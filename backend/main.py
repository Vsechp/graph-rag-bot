import os
import io
import re
import json
import traceback
import logging
import uuid
import time
import socket
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from functools import wraps
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from neo4j import GraphDatabase
from PIL import Image
from markdownify import markdownify as md

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
# Use LangChain agents for multi-agent system
from langchain.agents import AgentExecutor, create_openai_tools_agent
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback: simple character splitter
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function
            
            def split_text(self, text: str) -> List[str]:
                chunks = []
                i = 0
                text_len = len(text)
                while i < text_len:
                    end = min(i + self.chunk_size, text_len)
                    chunk = text[i:end].strip()
                    if chunk:
                        chunks.append(chunk)
                    if end >= text_len:
                        break
                    i = max(0, end - self.chunk_overlap)
                return chunks
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import PyPDF2
except ImportError:
    try:
        import pypdf
        PyPDF2 = pypdf
    except ImportError:
        PyPDF2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

# Load environment variables from .env file
load_dotenv()

# Metrics
try:
    from backend.metrics import (
        track_http_request,
        track_agent_invocation,
        track_llm_call,
        track_tool_call,
        track_neo4j_operation,
        update_neo4j_counts,
        track_guardrail_block,
        get_metrics,
        get_metrics_content_type
    )
except ImportError:
    # Fallback for Docker environment where we're in /app/backend/
    from metrics import (
        track_http_request,
        track_agent_invocation,
        track_llm_call,
        track_tool_call,
        track_neo4j_operation,
        update_neo4j_counts,
        track_guardrail_block,
        get_metrics,
        get_metrics_content_type
    )

# ==================== Observability Setup ====================

# Context for trace IDs
class TraceContext:
    def __init__(self):
        self.trace_id = None
    
    def set_trace_id(self, trace_id: str):
        self.trace_id = trace_id
    
    def get_trace_id(self) -> str:
        return self.trace_id or str(uuid.uuid4())

trace_context = TraceContext()

# Structured logging
class TraceIDFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = trace_context.get_trace_id()
        return True

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [trace_id=%(trace_id)s] - %(message)s',
    handlers=[
        logging.FileHandler('logs/backend.log'),
        logging.StreamHandler()
    ]
)

# Add trace ID filter
trace_filter = TraceIDFilter()
for handler in logging.root.handlers:
    handler.addFilter(trace_filter)

logger = logging.getLogger(__name__)

# Custom callback handler for observability
class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking agent execution"""
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.start_time = None
        self.tool_calls = []
        self.llm_calls = []
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        self.start_time = time.time()
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        duration = time.time() - self.start_time if self.start_time else 0
        
        # Try to get token usage from different sources
        token_usage = {}
        if hasattr(response, "llm_output") and response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})
        
        # Also check response_metadata if available
        if not token_usage and hasattr(response, "response_metadata"):
            token_usage = response.response_metadata.get("token_usage", {})
        
        # Check in kwargs
        if not token_usage and "token_usage" in kwargs:
            token_usage = kwargs["token_usage"]
        
        self.llm_calls.append({
            "duration": duration,
            "tokens": token_usage
        })
        
        # Track LLM metrics
        tokens_dict = {}
        if token_usage:
            tokens_dict = {
                "prompt_tokens": token_usage.get("prompt_tokens", 0),
                "completion_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0)
            }
        
        # Always track LLM call even if tokens are not available
        track_llm_call(model=LLM_MODEL, success=True, duration=duration, tokens=tokens_dict if tokens_dict else None)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        tool_name = serialized.get("name", "unknown")
        self.tool_calls.append({"tool": tool_name, "input": input_str, "start_time": time.time()})
    
    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self.tool_calls:
            last_call = self.tool_calls[-1]
            duration = time.time() - last_call.get("start_time", time.time())
            last_call["duration"] = duration
            last_call["output_length"] = len(output) if output else 0
            
            # Track tool metrics
            tool_name = last_call.get("tool", "unknown")
            success = not (output and (output.startswith("Error") or output.startswith("Invalid")))
            track_tool_call(tool_name=tool_name, success=success, duration=duration)

# Configuration - все значения из .env

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-30b-a3b-thinking-2507")

MAX_CHARS = int(os.getenv("MAX_CHARS"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
CONTEXT_CHAR_LIMIT = int(os.getenv("CONTEXT_CHAR_LIMIT"))

ENABLE_INPUT_FILTERING = os.getenv("ENABLE_INPUT_FILTERING").lower() == "true"
ENABLE_OUTPUT_FILTERING = os.getenv("ENABLE_OUTPUT_FILTERING").lower() == "true"
MAX_URL_LENGTH = int(os.getenv("MAX_URL_LENGTH"))
MAX_CONTENT_SIZE = int(os.getenv("MAX_CONTENT_SIZE"))

app = FastAPI(title="GraphRAG Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD
)

# Initialize LLM with OpenRouter
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    temperature=0.0,
    timeout=30,
    max_retries=1,
    model_kwargs={
        "extra_headers": {
            "HTTP-Referer": "https://github.com/graph-rag-bot",
            "X-Title": "GraphRAG Bot",
        }
    }
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=MAX_CHARS,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
)

# ==================== Guardrails ====================

def validate_url(url: str) -> bool:
    """Validate URL format and safety"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ["http", "https"]:
            return False
        if not parsed.netloc:
            return False
        if len(url) > MAX_URL_LENGTH:
            return False
        # Block localhost and private IPs
        if parsed.hostname in ["localhost", "127.0.0.1"] or parsed.hostname.startswith("192.168."):
            return False
        return True
    except Exception:
        return False

def filter_input_prompt(prompt: str) -> Tuple[bool, str]:
    """Filter input for prompt injection attempts"""
    if not ENABLE_INPUT_FILTERING:
        return True, prompt
    
    # Basic prompt injection patterns
    suspicious_patterns = [
        "ignore previous instructions",
        "forget everything",
        "you are now",
        "system:",
        "assistant:",
        "user:",
        "###",
        "---",
    ]
    
    prompt_lower = prompt.lower()
    for pattern in suspicious_patterns:
        if pattern in prompt_lower:
            return False, "Input contains potentially unsafe content"
    
    return True, prompt

def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    if not text:
        return text
    
    # Remove bold/italic markdown
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # *italic*
    text = re.sub(r'__([^_]+)__', r'\1', text)  # __bold__
    text = re.sub(r'_([^_]+)_', r'\1', text)  # _italic_
    text = re.sub(r'~~([^~]+)~~', r'\1', text)  # ~~strikethrough~~
    
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)  # ```code blocks```
    text = re.sub(r'`([^`]+)`', r'\1', text)  # `inline code`
    
    # Remove links but keep text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url)
    
    # Remove headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # # Header
    
    # Clean up extra spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces
    
    return text.strip()


def filter_output_content(content: str) -> Tuple[bool, str]:
    """Filter output for sensitive information and remove markdown"""
    # Always clean markdown formatting
    content = clean_markdown(content)
    
    if not ENABLE_OUTPUT_FILTERING:
        return True, content
    
    # Basic PII patterns (simplified)
    pii_patterns = [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    ]
    
    import re
    for pattern in pii_patterns:
        if re.search(pattern, content):
            # Replace with placeholder
            content = re.sub(pattern, "[REDACTED]", content)
    
    return True, content

# ==================== Retry Logic ====================

def retry_with_backoff(max_retries: int = 2, backoff_factor: float = 0.5):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_factor * (2 ** attempt)
                        logger.warning(f"Retry {attempt + 1}/{max_retries} after {wait_time}s: {str(e)}",
                                     extra={"trace_id": trace_context.get_trace_id()})
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All retries exhausted: {str(e)}",
                                    extra={"trace_id": trace_context.get_trace_id()})
            raise last_exception
        return wrapper
    return decorator

# ==================== Tools ====================

@tool
def fetch_url_content(url: str) -> str:
    """Fetch content from a URL and convert it to markdown format.
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        Markdown formatted content from the URL
    """
    trace_id = trace_context.get_trace_id()
    
    # Validate URL
    if not validate_url(url):
        error_msg = f"Invalid or unsafe URL: {url}"
        logger.error(error_msg, extra={"trace_id": trace_id})
        return error_msg
    
    # Filter input
    is_safe, filtered_url = filter_input_prompt(url)
    if not is_safe:
        return filtered_url
    
    @retry_with_backoff(max_retries=2, backoff_factor=0.5)
    def _fetch():
        # Try multiple User-Agent strings to avoid blocking
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ]
        
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=2)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        last_response = None
        last_error = None
        
        # Try with different User-Agents
        for i, user_agent in enumerate(user_agents):
            try:
                headers = {
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "DNT": "1",
                    "Referer": "https://www.google.com/"
                }
                
                session.headers.clear()
                session.headers.update(headers)
                
                # Add small delay between attempts
                if i > 0:
                    time.sleep(0.5)
                
                response = session.get(url, timeout=15, allow_redirects=True)
                last_response = response
                
                # If successful, break
                if response.status_code == 200:
                    break
                
                # If 403, try next User-Agent
                if response.status_code == 403:
                    logger.warning(f"Got 403 for {url} with User-Agent {i+1}/{len(user_agents)}, trying next...", extra={"trace_id": trace_id})
                    continue
                
                # For other non-200 status codes, raise
                if response.status_code != 200:
                    response.raise_for_status()
                    
            except requests.exceptions.HTTPError as e:
                last_error = e
                if e.response and e.response.status_code == 403:
                    logger.warning(f"HTTP 403 error for {url} with User-Agent {i+1}/{len(user_agents)}, trying next...", extra={"trace_id": trace_id})
                    continue
                else:
                    # For non-403 errors, re-raise to be handled by outer try-except
                    raise
        
        # Check if we got a successful response
        if last_response and last_response.status_code == 200:
            response = last_response
        elif last_response and last_response.status_code == 403:
            # All attempts failed with 403
            error_msg = f"Error fetching URL {url}: 403 Client Error: Forbidden. The website is blocking automated requests. Please try accessing the URL manually or contact support."
            logger.error(error_msg, extra={"trace_id": trace_id})
            return error_msg
        elif last_error:
            # Re-raise the last error if it wasn't 403
            raise last_error
        else:
            # No response at all
            error_msg = f"Error fetching URL {url}: Failed to get response after multiple attempts"
            logger.error(error_msg, extra={"trace_id": trace_id})
            return error_msg
        
        # If we get here, we have a successful response (status 200)
        # No need to call raise_for_status() since we already verified status_code == 200
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Remove only truly unwanted elements (scripts, styles, etc.)
        for tag in soup(["script", "style", "noscript", "iframe", "embed", "object"]):
            tag.decompose()
        
        # Universal content extraction: use body for all sites
        main_content = soup.find("body")
        if not main_content:
            main_content = soup
        
        html_content = str(main_content) if main_content else str(soup)
        
        # Convert to markdown
        markdown_content = md(html_content, heading_style="ATX")
        
        # Limit size intelligently - keep first part which usually contains most important info
        # Increase limit to allow more content (500KB = 500000 chars)
        max_chars = MAX_CONTENT_SIZE * 10  # Allow 500KB of markdown text for large pages
        original_length = len(markdown_content)
        if len(markdown_content) > max_chars:
            # Try to truncate at a paragraph boundary
            truncated = markdown_content[:max_chars]
            last_paragraph = truncated.rfind('\n\n')
            if last_paragraph > max_chars * 0.8:  # If we can find a paragraph break in last 20%
                markdown_content = truncated[:last_paragraph] + "\n\n[Content truncated due to size...]"
            else:
                markdown_content = truncated + "\n\n[Content truncated due to size...]"
        
        
        return markdown_content
    
    try:
        return _fetch()
    except requests.exceptions.ConnectionError as e:
        error_msg = f"DNS/Connection error fetching URL {url}: {str(e)}. Please check network connectivity and DNS settings."
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        # Try to provide more helpful error message
        if "Name or service not known" in str(e) or "Errno -2" in str(e):
            error_msg = f"DNS resolution failed for {url}. The domain name could not be resolved. Please check if the URL is correct and the DNS server is accessible."
        return error_msg
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout error fetching URL {url}: {str(e)}"
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg
    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching URL {url}: {str(e)}"
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg
    except Exception as e:
        error_msg = f"Unexpected error fetching URL: {str(e)}"
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg


# Semantic RAG Tools (following the article approach)
def extract_doc_type_from_question(question: str) -> Optional[str]:
    """Extract document type from question (pdf, image, webpage, file).
    
    Args:
        question: User question
        
    Returns:
        Document type (pdf, image, webpage, file) or None
    """
    question_lower = question.lower()
    
    # PDF keywords
    pdf_keywords = ["pdf", "пдф", "pdf файл", "пдф файл", "pdf документ", "пдф документ", 
                    "pdf файл который", "пдф файл который", "pdf который", "пдф который"]
    if any(keyword in question_lower for keyword in pdf_keywords):
        return "pdf"
    
    # Image keywords
    image_keywords = ["картинк", "изображен", "фото", "picture", "image", "img", 
                      "картинк которую", "изображение которое", "фото которое"]
    if any(keyword in question_lower for keyword in image_keywords):
        return "image"
    
    # Webpage/site keywords
    webpage_keywords = ["сайт", "страниц", "веб-страниц", "webpage", "website", "site",
                       "сайт который", "страница которую", "этот сайт", "эта страница"]
    if any(keyword in question_lower for keyword in webpage_keywords):
        return "webpage"
    
    # Generic file keywords
    file_keywords = ["файл", "документ", "file", "document",
                    "файл который", "документ который", "файл что", "документ что"]
    if any(keyword in question_lower for keyword in file_keywords):
        return "file"
    
    return None


def extract_source_from_question(question: str, available_sources: List[Dict[str, Any]]) -> Optional[str]:
    """Extract source mention from question by matching against available sources.
    
    Args:
        question: User question
        available_sources: List of available document sources
        
    Returns:
        Matched source URL or None
    """
    question_lower = question.lower()
    
    # Check for direct URL mentions
    url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
    urls = url_pattern.findall(question)
    if urls:
        # Try to match against available sources
        for url in urls:
            normalized_url = url.lower().rstrip('/')
            for doc in available_sources:
                doc_source = doc.get("source", "").lower().rstrip('/')
                if normalized_url == doc_source or normalized_url in doc_source or doc_source in normalized_url:
                    return doc.get("source")
    
    # Check for domain mentions (bbc, bbc.com, etc.)
    for doc in available_sources:
        source = doc.get("source", "").lower()
        # Extract domain from source
        try:
            parsed = urlparse(source if source.startswith("http") else f"https://{source}")
            domain = parsed.netloc or parsed.path.split('/')[0]
            domain_parts = domain.split('.')
            
            # Check if any domain part is mentioned in question
            for part in domain_parts:
                if len(part) > 2 and part in question_lower:
                    return doc.get("source")
            
            # Check if full domain is mentioned
            if domain in question_lower or domain.replace('.', ' ') in question_lower:
                return doc.get("source")
        except:
            # If parsing fails, check if source string is in question
            source_clean = source.replace('https://', '').replace('http://', '').rstrip('/')
            if source_clean in question_lower or any(part in question_lower for part in source_clean.split('.') if len(part) > 2):
                return doc.get("source")
    
    return None


@tool
def vector_search(anchor: str, label: str, query: str, limit: int = 10, source_filter: Optional[str] = None) -> str:
    """Vector search for semantic similarity. Searches chunks by semantic meaning.
    
    Args:
        anchor: Document type to search in (e.g., "chunk")
        label: Attribute to search (e.g., "text")
        query: Search query
        limit: Maximum results
        source_filter: Optional source URL to filter by
        
    Returns:
        Found chunks with source information
    """
    trace_id = trace_context.get_trace_id()
    
    try:
        with driver.session() as session:
            # Build query with optional source filter
            if source_filter:
                # When source is filtered, get all chunks from that source (for questions like "о чем этот сайт")
                search_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WHERE d.source = $source_filter
                RETURN c.text AS text, d.source AS source, d.type AS doc_type, 
                       c.chunk_index AS idx, d.created_at AS created_at
                ORDER BY 
                    d.created_at DESC, 
                    idx ASC
                LIMIT $limit
                """
                result = session.run(search_query, source_filter=source_filter, limit=limit)
            else:
                # Use text search with semantic-like matching (for true vector search would need embeddings)
                # Search for chunks containing query terms or semantically related
                search_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WHERE toLower(c.text) CONTAINS toLower($query)
                RETURN c.text AS text, d.source AS source, d.type AS doc_type, 
                       c.chunk_index AS idx, d.created_at AS created_at
                ORDER BY 
                    CASE WHEN toLower(c.text) CONTAINS toLower($query) THEN 1 ELSE 2 END,
                    d.created_at DESC, 
                    idx ASC
                LIMIT $limit
                """
                result = session.run(search_query, query=query, limit=limit)
            
            chunks = []
            for record in result:
                text = record.get("text", "")
                source = record.get("source", "")
                doc_type = record.get("doc_type", "webpage")
                created_at = record.get("created_at", "")
                if text and text.strip():
                    chunks.append({
                        "text": text.strip(),
                        "source": source,
                        "doc_type": doc_type,
                        "created_at": created_at
                    })
            
            if chunks:
                result_parts = []
                for chunk_info in chunks:
                    source_info = f"[Source: {chunk_info['source']}, Type: {chunk_info['doc_type']}]"
                    result_parts.append(f"{source_info}\n{chunk_info['text']}")
                return "\n\n---\n\n".join(result_parts)
            else:
                return "No chunks found with vector search."
    except Exception as e:
        logger.warning(f"Vector search failed: {e}", extra={"trace_id": trace_id})
        return "No chunks found."


@tool
def cypher_query(query: str) -> str:
    """Execute a Cypher query for structured data retrieval.
    
    Args:
        query: Cypher query string
        
    Returns:
        Query results as formatted text
    """
    trace_id = trace_context.get_trace_id()
    
    try:
        with driver.session() as session:
            result = session.run(query)
            records = []
            for record in result:
                # Format record as key-value pairs
                record_dict = dict(record)
                records.append(record_dict)
            
            if records:
                # Format results
                result_parts = []
                for rec in records[:20]:  # Limit to 20 records
                    parts = [f"{k}: {v}" for k, v in rec.items() if v is not None]
                    result_parts.append(" | ".join(parts))
                return "\n".join(result_parts)
            else:
                return "Query returned no results."
    except Exception as e:
        error_msg = f"Cypher query error: {str(e)}"
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg


@tool
def search_graph_rag(question: str, limit: int = 20) -> str:
    """Search the knowledge graph using LLM-based semantic understanding.
    Uses LLM to understand the question semantically and find relevant chunks.
    
    Args:
        question: The question to search for
        limit: Maximum number of chunks to return
        
    Returns:
        Relevant text chunks from the knowledge graph
    """
    trace_id = trace_context.get_trace_id()
    search_start = time.time()
    
    # Filter input
    is_safe, filtered_question = filter_input_prompt(question)
    if not is_safe:
        track_guardrail_block("prompt_injection")
        return filtered_question
    
    try:
        with driver.session() as session:
            # Step 1: Get all available documents
            schema_query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
            WITH d, count(c) as chunk_count
            RETURN d.source as source, d.type as doc_type, d.created_at as created_at, chunk_count
            ORDER BY d.created_at DESC
            LIMIT 50
            """
            schema_result = session.run(schema_query)
            available_docs = []
            for record in schema_result:
                available_docs.append({
                    "source": record.get("source", ""),
                    "type": record.get("doc_type", "webpage"),
                    "created_at": record.get("created_at", ""),
                    "chunk_count": record.get("chunk_count", 0)
                })
            
            if not available_docs:
                track_neo4j_operation("search", success=False, duration=time.time() - search_start)
                return "No documents found in the knowledge graph."
            
            # Step 2: Extract source from question if explicitly mentioned (for filtering)
            # Only filter by source if it's explicitly mentioned (URL or domain), not by document type
            matched_source = extract_source_from_question(filtered_question, available_docs)
            
            # Step 3: Get chunks from database (semantic search - no keyword filtering)
            # LLM will understand context and select relevant chunks based on document type, source, etc.
            if matched_source:
                # Only filter by source if explicitly mentioned (URL/domain in question)
                chunks_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                WHERE d.source = $source
                RETURN c.text AS text, d.source AS source, d.type AS doc_type, 
                       c.chunk_index AS idx, d.created_at AS created_at
                ORDER BY d.created_at DESC, idx ASC
                LIMIT 200
                """
                chunks_result = session.run(chunks_query, source=matched_source)
            else:
                # Get chunks from all documents - LLM will semantically understand which ones are relevant
                chunks_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.text AS text, d.source AS source, d.type AS doc_type, 
                       c.chunk_index AS idx, d.created_at AS created_at
                ORDER BY d.created_at DESC, idx ASC
                LIMIT 200
                """
                chunks_result = session.run(chunks_query)
            
            # Collect all chunks
            all_chunks = []
            for record in chunks_result:
                text = record.get("text", "")
                source = record.get("source", "")
                doc_type = record.get("doc_type", "webpage")
                created_at = record.get("created_at", "")
                if text and text.strip():
                    all_chunks.append({
                        "text": text.strip(),
                        "source": source,
                        "doc_type": doc_type,
                        "created_at": created_at
                    })
            
            if not all_chunks:
                track_neo4j_operation("search", success=False, duration=time.time() - search_start)
                return "No chunks found in the knowledge graph."
            
            # Step 4: Use LLM to semantically understand the question and filter relevant chunks
            chunks_for_llm = []
            for idx, chunk_info in enumerate(all_chunks[:150]):  # Limit to 150 chunks for LLM processing
                chunks_for_llm.append(f"[Chunk {idx}]\nSource: {chunk_info['source']}\nType: {chunk_info['doc_type']}\nContent: {chunk_info['text'][:500]}")
            
            semantic_filter_prompt = f"""You are a semantic relevance filter. Your task is to understand the user's question semantically and select the most relevant chunks that can answer it.

User Question: {filtered_question}

Available Chunks:
{chr(10).join(chunks_for_llm)}

Your task is to understand the SEMANTIC meaning of the question and select relevant chunks:

1. Understand the FULL CONTEXT of the question:
   - What type of document is the user asking about? (PDF, image, webpage, file)
   - What source/document is mentioned? (URL, domain, filename, "этот сайт", "эта статья", "который я скинул ранее")
   - What temporal context? ("ранее", "earlier", "last", "последний")
   - What is the actual information being requested?

2. For each chunk, analyze:
   - Does the chunk's Source match what the user is asking about? (semantically, not just keyword matching)
   - Does the chunk's Type match the document type mentioned in the question? (if mentioned)
   - Is the chunk's Content relevant to answering the question?
   - Consider temporal context: if question mentions "ранее" or "earlier", prioritize chunks from older documents (lower created_at)

3. Semantic understanding examples:
   - "о чем пдф файл который я скинул ранее" → Find chunks from PDF documents (Type: pdf), prioritize older ones
   - "о чем этот сайт" → Find chunks from webpage documents (Type: webpage) matching the most recent or mentioned site
   - "что в картинке которую я отправил" → Find chunks from image documents (Type: image), most recent
   - "расскажи про php.net" → Find chunks from documents with source containing "php.net"

4. Select ONLY chunks that are semantically relevant to answering the question
5. Order by relevance: most relevant first, considering source match, type match, content relevance, and temporal context
6. Return ONLY a JSON array of chunk indices (0-based), ordered by relevance

Return format: [0, 5, 12, 3] (just the array, no explanation, no markdown)

If no chunks are relevant, return an empty array: []"""

            filter_messages = [
                SystemMessage(content="You are a semantic relevance filter. Understand questions semantically and select relevant chunks. Return only a JSON array of chunk indices."),
                HumanMessage(content=semantic_filter_prompt)
            ]
            
            filter_start = time.time()
            filter_response = llm.invoke(filter_messages)
            filter_duration = time.time() - filter_start
            filter_result = filter_response.content if hasattr(filter_response, 'content') else str(filter_response)
            
            # Track LLM call
            tokens_dict = None
            if hasattr(filter_response, 'response_metadata') and filter_response.response_metadata:
                usage = filter_response.response_metadata.get('token_usage', {})
                if usage:
                    tokens_dict = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
            track_llm_call(model=LLM_MODEL, success=True, duration=filter_duration, tokens=tokens_dict)
            
            # Parse filter result
            try:
                filter_clean = filter_result.strip()
                if "```json" in filter_clean:
                    filter_clean = filter_clean.split("```json")[1].split("```")[0].strip()
                elif "```" in filter_clean:
                    filter_clean = filter_clean.split("```")[1].split("```")[0].strip()
                
                json_start = filter_clean.find("[")
                json_end = filter_clean.rfind("]") + 1
                if json_start >= 0 and json_end > json_start:
                    relevant_indices = json.loads(filter_clean[json_start:json_end])
                else:
                    relevant_indices = json.loads(filter_clean)
                
                if relevant_indices and len(relevant_indices) > 0:
                    filtered_chunks = [all_chunks[i] for i in relevant_indices if 0 <= i < len(all_chunks)]
                    chunks_with_source = filtered_chunks[:limit]
                else:
                    # If LLM found nothing, return first chunks as fallback
                    chunks_with_source = all_chunks[:limit]
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                logger.warning(f"Failed to parse LLM filter response: {filter_result[:200]}, using all chunks", extra={"trace_id": trace_id})
                chunks_with_source = all_chunks[:limit]
            
            if not chunks_with_source:
                track_neo4j_operation("search", success=False, duration=time.time() - search_start)
                return "No relevant information found in the knowledge graph."
            
            # Format results with source information
            result_parts = []
            for chunk_info in chunks_with_source:
                source_info = f"[Source: {chunk_info['source']}, Type: {chunk_info['doc_type']}]"
                result_parts.append(f"{source_info}\n{chunk_info['text']}")
            result_text = "\n\n---\n\n".join(result_parts)
            track_neo4j_operation("search", success=True, duration=time.time() - search_start)
            return result_text
            
    except Exception as e:
        error_msg = f"Error searching graph: {str(e)}"
        track_neo4j_operation("search", success=False, duration=time.time() - search_start)
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg


@tool
def store_document(source: str, content: str, doc_type: str = "webpage") -> str:
    """Store a document and its chunks in the knowledge graph.
    
    Args:
        source: Source identifier (URL or filename)
        content: Document content to store (must be the actual markdown content, not a placeholder)
        doc_type: Type of document (webpage, pdf, image, etc.)
        
    Returns:
        Confirmation message
    """
    trace_id = trace_context.get_trace_id()
    
    # Validate content is not a placeholder
    if not content or content.strip() in ["MARKDOWN_CONTENT_FROM_FETCH", "MARKDOWN_CONTENT", "content", ""]:
        error_msg = f"Invalid content: received placeholder or empty content. Content must be the actual markdown text from fetch_url_content."
        logger.error(error_msg, extra={"trace_id": trace_id})
        return error_msg
    
    if len(content) < 50:
        error_msg = f"Content too short ({len(content)} chars). Expected actual markdown content from fetch_url_content."
        logger.error(error_msg, extra={"trace_id": trace_id})
        return error_msg
    
    try:
        created_at = datetime.utcnow().isoformat() + "Z"
    
        # Basic content cleaning - remove only empty lines
        filtered_content = content
        lines = filtered_content.split('\n')
        filtered_lines = [line for line in lines if line.strip()]
        filtered_content = '\n'.join(filtered_lines).strip()
        
        # Split into chunks
        chunks = text_splitter.split_text(filtered_content) if filtered_content and filtered_content.strip() else []
        if not chunks:
            if filtered_content and filtered_content.strip():
                chunks = [filtered_content.strip()]
            else:
                logger.warning(f"No content to chunk after filtering. Original content length: {len(content) if content else 0}", extra={"trace_id": trace_id})
                chunks = []
        
        # Filter out empty or very short chunks (less than 20 characters)
        # But be less aggressive - keep chunks that are at least 10 characters
        original_chunk_count = len(chunks)
        chunks = [chunk for chunk in chunks if chunk and chunk.strip() and len(chunk.strip()) >= 10]
        
        
        # If all chunks were filtered out, use the original content if it's not empty
        if not chunks:
            if filtered_content and filtered_content.strip() and len(filtered_content.strip()) >= 10:
                logger.warning("All chunks filtered out, using full filtered content as single chunk", extra={"trace_id": trace_id})
                chunks = [filtered_content.strip()]
            elif content and content.strip() and len(content.strip()) >= 10:
                logger.warning("Using original content as fallback (filtered content was empty)", extra={"trace_id": trace_id})
                chunks = [content.strip()]
            else:
                logger.error(f"No valid content to store. Filtered: {len(filtered_content) if filtered_content else 0} chars, Original: {len(content) if content else 0} chars", extra={"trace_id": trace_id})
                chunks = []
        
        saved_chunks = 0
        neo4j_start = time.time()
        with driver.session() as session:
            with session.begin_transaction() as tx:
                # Create document
                doc_start = time.time()
                result = tx.run(
                    """
                    CREATE (d:Document {source: $source, type: $type, created_at: $created_at})
                    RETURN elementId(d) AS doc_id
                    """,
                    source=source,
                    type=doc_type,
                    created_at=created_at
                )
                record = result.single()
                doc_id = record["doc_id"]
                track_neo4j_operation("create_document", success=True, duration=time.time() - doc_start)
                
                # Create chunks in batch for better performance
                if not chunks:
                    logger.error(f"No chunks to store for document {source}. Document created but no chunks.", extra={"trace_id": trace_id})
                    return f"Error: Document created but no chunks to store. Content may have been filtered out completely."
                
                # Prepare chunk data for batch operation
                chunk_data = [
                    {
                        "text": chunk_content.strip(),
                        "chunk_index": idx,
                        "created_at": created_at
                    }
                    for idx, chunk_content in enumerate(chunks)
                    if chunk_content and chunk_content.strip()
                ]
                
                if not chunk_data:
                    logger.warning(f"No valid chunks to store after filtering", extra={"trace_id": trace_id})
                    return f"Error: No valid chunks to store after filtering."
                
                # Batch create all chunks in one query
                chunk_start = time.time()
                try:
                    result = tx.run(
                        """
                        UNWIND $chunks AS chunk_data
                        MATCH (d) WHERE elementId(d) = $doc_id
                        CREATE (c:Chunk {
                            text: chunk_data.text,
                            chunk_index: chunk_data.chunk_index,
                            created_at: chunk_data.created_at
                        })
                        CREATE (d)-[:HAS_CHUNK {idx: chunk_data.chunk_index}]->(c)
                        RETURN count(c) AS created_count
                        """,
                        chunks=chunk_data,
                        doc_id=doc_id
                    )
                    record = result.single()
                    saved_chunks = record["created_count"] if record else len(chunk_data)
                    track_neo4j_operation("create_chunk", success=True, duration=time.time() - chunk_start)
                except Exception as chunk_error:
                    track_neo4j_operation("create_chunk", success=False, duration=time.time() - chunk_start)
                    logger.error(f"Error creating chunks in batch: {chunk_error}", extra={"trace_id": trace_id}, exc_info=True)
                    raise
                
                tx.commit()
        
        track_neo4j_operation("store_document", success=saved_chunks > 0, duration=time.time() - neo4j_start)
        
        if saved_chunks == 0:
            logger.error(f"No chunks were saved for document {source} despite having {len(chunks)} chunks to process", extra={"trace_id": trace_id})
            return f"Error: No chunks were saved. Please check logs for details."
        
        logger.info(f"Document stored: {source}. ({saved_chunks}/{len(chunks)} chunks saved)", extra={"trace_id": trace_id})
        return f"Document stored successfully. Created {saved_chunks} chunks."
    except Exception as e:
        error_msg = f"Error storing document: {str(e)}"
        logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
        return error_msg


# List of available tools
tools = [fetch_url_content, search_graph_rag, store_document, vector_search, cypher_query]

# ==================== Agents ====================

def create_agent_executor(agent_tools, system_prompt):
    """Create agent executor with proper tool call handling."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_openai_tools_agent(llm, agent_tools, prompt)
    # Reduced max_iterations to prevent long runs
    # Timeout is handled at the thread level in ingest_url
    executor = AgentExecutor(
        agent=agent, 
        tools=agent_tools, 
        verbose=False, 
        handle_parsing_errors=True, 
        max_iterations=3,  # Reduced from 5 to prevent long runs
        early_stopping_method="force"
    )
    return executor


def create_url_agent():
    """Create an agent specialized in URL processing."""
    system_prompt = """Process URL in 2 steps:
1. Call fetch_url_content(url) - returns markdown text
2. Call store_document(source=url, content=<exact_string_from_step_1>, doc_type="webpage")

RULES:
- Pass the EXACT string from fetch_url_content to store_document's content parameter
- Do NOT modify, summarize, or truncate the content
- Always use doc_type="webpage"
- Work quickly and efficiently

If fetch_url_content returns an error, return that error. Otherwise, call store_document with the exact content."""
    url_tools = [fetch_url_content, store_document]
    return create_agent_executor(url_tools, system_prompt)


def create_search_agent():
    """Create an agent specialized in searching the knowledge graph."""
    system_prompt = """You are a search agent. Your task:
1. Use search_graph_rag tool to find information
2. Answer the question based on search results

Use Russian if question is in Russian, English if in English. Be concise and complete. Use plain text only - do not use markdown formatting like **, *, __, #, `, or ```."""
    search_tools = [search_graph_rag]
    return create_agent_executor(search_tools, system_prompt)


def create_coordinator_agent():
    """Create a coordinator agent that determines user intent and routes to appropriate agents."""
    system_prompt = """You are a coordinator agent that determines user intent and routes requests to appropriate specialized agents.

Your task is to analyze user input and determine their intent. You must respond ONLY with valid JSON, no other text.

Intent types:
- SAVE_URL: User wants to save a web page (contains URL or commands like "сохрани страницу", "save page", "сохранить страницу")
- SEARCH: User wants to search/query the knowledge graph (asks a question about saved data, mentions previously saved sources like "bbc", "bbc.com", "этот сайт", "эта статья", "который я скинул ранее")
- UNKNOWN: Intent is unclear

Rules:
1. If input contains a URL (http:// or https://) or save commands ("сохрани страницу", "save page"), intent is SAVE_URL
2. If input is a question (contains "?", "о чем", "что такое", "расскажи", "what is", "tell me", "который я скинул", "ранее", "этот сайт", "эта статья", mentions domain names like "bbc", "bbc.com"), intent is SEARCH
3. Extract URL from input if present
4. For SAVE_URL, extract the URL from the input text
5. For SEARCH, use the original input as the query (even if it mentions a previously saved source)

You MUST respond with ONLY this JSON format (no markdown, no code blocks, no explanation):
{{"intent": "SAVE_URL" | "SEARCH" | "UNKNOWN", "url": "<extracted URL or null>", "query": "<search query or null>", "reasoning": "<brief explanation>"}}"""
    
    coordinator_tools = []
    return create_agent_executor(coordinator_tools, system_prompt)


def determine_user_intent(user_input: str, trace_id: str) -> Dict[str, Any]:
    """Determine user intent using coordinator agent."""
    try:
        result = coordinator_agent.invoke({"input": user_input})
        output = result.get("output", str(result)) if isinstance(result, dict) else str(result)
        
        # Extract JSON from response (may contain markdown code blocks or extra text)
        output_clean = output.strip()
        
        # Remove markdown code blocks if present
        if "```json" in output_clean:
            output_clean = output_clean.split("```json")[1].split("```")[0].strip()
        elif "```" in output_clean:
            output_clean = output_clean.split("```")[1].split("```")[0].strip()
        
        # Try to find JSON object in the response
        json_start = output_clean.find("{")
        json_end = output_clean.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = output_clean[json_start:json_end]
            intent_result = json.loads(json_str)
            
            # Validate and normalize the result
            intent = intent_result.get("intent", "SEARCH").upper()
            if intent not in ["SAVE_URL", "SEARCH", "UNKNOWN"]:
                intent = "SEARCH"
            
            return {
                "intent": intent,
                "url": intent_result.get("url"),
                "query": intent_result.get("query"),
                "reasoning": intent_result.get("reasoning", "Determined by coordinator agent")
            }
        else:
            # Fallback: try to parse as JSON directly
            intent_result = json.loads(output_clean)
            return {
                "intent": intent_result.get("intent", "SEARCH").upper(),
                "url": intent_result.get("url"),
                "query": intent_result.get("query"),
                "reasoning": intent_result.get("reasoning", "Determined by coordinator agent")
            }
            
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse coordinator response as JSON: {output[:200] if 'output' in locals() else 'N/A'}", extra={"trace_id": trace_id})
        # Fallback: simple heuristic
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
        urls = url_pattern.findall(user_input)
        user_input_lower = user_input.lower()
        save_keywords = ["сохрани страницу", "save page", "сохранить страницу"]
        has_save = any(kw in user_input_lower for kw in save_keywords)
        
        if urls or has_save:
            return {
                "intent": "SAVE_URL",
                "url": urls[0] if urls else None,
                "query": None,
                "reasoning": "Fallback: URL or save command detected"
            }
        else:
            return {
                "intent": "SEARCH",
                "url": None,
                "query": user_input,
                "reasoning": "Fallback: defaulting to search"
            }
    except Exception as e:
        logger.error(f"Error determining intent: {e}", extra={"trace_id": trace_id}, exc_info=True)
        return {
            "intent": "SEARCH",
            "url": None,
            "query": user_input,
            "reasoning": "Error in coordinator agent, defaulting to search"
        }


# Initialize agents
url_agent = create_url_agent()
search_agent = create_search_agent()
coordinator_agent = create_coordinator_agent()

# Middleware

@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """Add trace ID to each request for distributed tracing"""
    # Distributed tracing: accept trace ID from upstream (bot) or generate new one
    trace_id = (
        request.headers.get("X-Trace-Id") or 
        request.headers.get("Trace-Id") or 
        request.headers.get("X-Request-Id") or
        str(uuid.uuid4())
    )
    trace_context.set_trace_id(trace_id)
    
    # Add trace ID to request state
    request.state.trace_id = trace_id
    
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Track HTTP request metrics
        method = request.method
        endpoint = request.url.path
        status = response.status_code
        track_http_request(method=method, endpoint=endpoint, status=status, duration=duration)
        
        # Propagate trace ID to downstream services
        response.headers["X-Trace-Id"] = trace_id
        response.headers["Trace-Id"] = trace_id
        response.headers["X-Request-Duration"] = f"{duration:.3f}"
        
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s",
                   extra={"trace_id": trace_id})
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        method = request.method
        endpoint = request.url.path
        track_http_request(method=method, endpoint=endpoint, status=500, duration=duration)
        logger.error(f"Request failed: {str(e)} - {duration:.3f}s", 
                    extra={"trace_id": trace_id}, exc_info=True)
        raise

# ==================== Helper Functions ====================

def extract_text_from_html(url: str) -> str:
    """Extract text from HTML URL."""
    try:
        result = fetch_url_content.invoke({"url": url})
        if result.startswith("Error"):
            raise Exception(result)
        return result
    except Exception as e:
        raise Exception(f"Failed to fetch URL: {e}")


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber with PyPDF2 fallback."""
    trace_id = trace_context.get_trace_id()
    
    if not data or len(data) == 0:
        error_msg = "PDF data is empty"
        logger.error(error_msg, extra={"trace_id": trace_id})
        raise Exception(error_msg)
    
    
    # Try pdfplumber first (better quality)
    if pdfplumber:
        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text_parts = []
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    raise Exception("PDF has no pages")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_parts.append(page_text.strip())
                        # Progress logging removed for cleaner logs
                    except Exception as page_error:
                        logger.warning(f"Error extracting text from page {i + 1}: {page_error}", extra={"trace_id": trace_id})
                        continue
                
                if text_parts:
                    result = "\n".join(text_parts)
                    return result
                else:
                    logger.warning("pdfplumber extracted no text, trying PyPDF2 fallback", extra={"trace_id": trace_id})
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2 fallback", extra={"trace_id": trace_id})
    
    # Fallback to PyPDF2/pypdf
    if PyPDF2:
        try:
            pdf_file = io.BytesIO(data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            total_pages = len(pdf_reader.pages)
            
            if total_pages == 0:
                raise Exception("PDF has no pages")
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception as page_error:
                    logger.warning(f"Error extracting text from page {i + 1}: {page_error}", extra={"trace_id": trace_id})
                    continue
            
            if text_parts:
                result = "\n".join(text_parts)
                return result
        except Exception as e:
            logger.error(f"PyPDF2 fallback also failed: {e}", extra={"trace_id": trace_id}, exc_info=True)
    
    # If both methods failed
    error_msg = "No text could be extracted from PDF. The file may be image-based, encrypted, or corrupted. Tried pdfplumber and PyPDF2."
    logger.error(error_msg, extra={"trace_id": trace_id})
    raise Exception(error_msg)


def extract_text_from_image_bytes(data: bytes) -> str:
    """Extract text from image bytes using OCR."""
    if not pytesseract:
        raise Exception("pytesseract not installed")
    
    try:
        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(image, lang="rus+eng")
    except Exception as e:
        raise Exception(f"OCR failed: {e}")


def create_fulltext_index():
    """Create fulltext index for chunks."""
    cypher = "CALL db.index.fulltext.createNodeIndex('chunks_text_idx', ['Chunk'], ['text'])"
    with driver.session() as session:
        try:
            session.run(cypher)
        except Exception:
            pass
# ==================== API Models ====================

class IngestURL(BaseModel):
    url: str = Field(..., max_length=MAX_URL_LENGTH)
    
    @validator('url')
    def validate_url(cls, v):
        if not validate_url(v):
            raise ValueError("Invalid or unsafe URL")
        return v


class Question(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    
    @validator('question')
    def validate_question(cls, v):
        is_safe, _ = filter_input_prompt(v)
        if not is_safe:
            raise ValueError("Question contains potentially unsafe content")
        return v


class TimeQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    since: Optional[str] = None
    until: Optional[str] = None
    
    @validator('question')
    def validate_question(cls, v):
        is_safe, _ = filter_input_prompt(v)
        if not is_safe:
            raise ValueError("Question contains potentially unsafe content")
        return v

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    try:
        create_fulltext_index()
        graph.refresh_schema()
        logger.info("Backend started successfully")
    except Exception as e:
        logger.warning(f"Startup warning: {e}", exc_info=True)


@app.post("/ingest/url")
def ingest_url(payload: IngestURL, request: Request):
    """Ingest URL using URL agent with fallback to direct tool calls."""
    trace_id = getattr(request.state, "trace_id", trace_context.get_trace_id())
    trace_context.set_trace_id(trace_id)
    
    try:
        logger.info(f"Ingesting URL: {payload.url}", extra={"trace_id": trace_id})
        
        # Try URL agent first (following multi-agent architecture)
        agent_start_time = time.time()
        agent_result = None
        agent_error = None
        output = None
        
        def run_agent():
            nonlocal agent_result, agent_error
            try:
                agent_result = url_agent.invoke({"input": f"Fetch and store this URL: {payload.url}"})
            except Exception as e:
                agent_error = e
        
        # Run agent in thread with timeout
        agent_thread = threading.Thread(target=run_agent, daemon=True)
        agent_thread.start()
        agent_thread.join(timeout=90)  # 90 seconds timeout
        
        agent_duration = time.time() - agent_start_time
        
        if agent_thread.is_alive():
            # Agent timed out - use fallback
            track_agent_invocation("url_agent", success=False, duration=agent_duration)
            logger.warning(f"URL agent timed out after 90s, using direct tool calls", extra={"trace_id": trace_id})
        elif agent_error:
            # Agent raised exception - use fallback
            track_agent_invocation("url_agent", success=False, duration=agent_duration)
            logger.warning(f"URL agent error: {type(agent_error).__name__}, using direct tool calls", extra={"trace_id": trace_id})
        elif agent_result:
            # Agent completed - check result
            output = agent_result.get("output", str(agent_result)) if isinstance(agent_result, dict) else str(agent_result)
            
            if output and not output.strip().startswith("Error") and len(output.strip()) > 10:
                # Agent succeeded
                track_agent_invocation("url_agent", success=True, duration=agent_duration)
                logger.info(f"URL processed successfully by url_agent", extra={"trace_id": trace_id})
            else:
                # Agent returned error or empty - use fallback
                track_agent_invocation("url_agent", success=False, duration=agent_duration)
                logger.warning(f"URL agent returned error or empty output, using direct tool calls", extra={"trace_id": trace_id})
                output = None
        else:
            # No result from agent - use fallback
            track_agent_invocation("url_agent", success=False, duration=agent_duration)
            logger.warning(f"URL agent returned no result, using direct tool calls", extra={"trace_id": trace_id})
        
        # Fallback to direct tool calls if agent failed
        if not output:
            logger.debug(f"Processing URL using direct tool calls (fallback)", extra={"trace_id": trace_id})
            tool_start = time.time()
            content = fetch_url_content.invoke({"url": payload.url})
            track_tool_call("fetch_url_content", success=not (content.startswith("Error") or content.startswith("Invalid")), duration=time.time() - tool_start)
            
            if content.startswith("Error") or content.startswith("Invalid"):
                logger.error(f"Failed to fetch URL content: {content[:200]}", extra={"trace_id": trace_id})
                return {"status": "error", "message": content}
            
            if not content or len(content.strip()) < 50:
                logger.error(f"Content too short or empty: {len(content) if content else 0} chars", extra={"trace_id": trace_id})
                return {"status": "error", "message": "Failed to extract meaningful content from URL"}
            
            tool_start = time.time()
            result = store_document.invoke({
                "source": payload.url,
                "content": content,
                "doc_type": "webpage"
            })
            track_tool_call("store_document", success=not result.startswith("Error"), duration=time.time() - tool_start)
            output = result
        
        if output.startswith("Error"):
            logger.error(f"Failed to process URL: {output[:200]}", extra={"trace_id": trace_id})
            return {"status": "error", "message": output}
        
        # Verify document was stored
        with driver.session() as session:
            check_query = """
            MATCH (d:Document {source: $source})-[:HAS_CHUNK]->(c:Chunk)
            RETURN count(c) AS chunk_count
            """
            check_result = session.run(check_query, source=payload.url)
            record = check_result.single()
            chunk_count = record["chunk_count"] if record else 0
            
            if chunk_count == 0:
                return {"status": "error", "message": "URL processed but no chunks were saved to database"}
        
        logger.info(f"Successfully ingested URL: {payload.url}, created {chunk_count} chunks", extra={"trace_id": trace_id})
        return {"status": "ok", "source": payload.url, "message": output}
    except Exception as e:
        logger.error(f"Ingest URL error: {e}", extra={"trace_id": trace_id}, exc_info=True)
        return {"status": "error", "message": str(e)[:200]}


@app.post("/ingest/file")
def ingest_file(file: UploadFile = File(...), request: Request = None):
    """Ingest file (PDF, image, etc.) with observability."""
    trace_id = getattr(request.state, "trace_id", trace_context.get_trace_id()) if request else trace_context.get_trace_id()
    trace_context.set_trace_id(trace_id)
    
    try:
        logger.info(f"Ingesting file: {file.filename}, content_type: {file.content_type}", extra={"trace_id": trace_id})
        
        # Read file data - ensure we read from the beginning
        data = file.file.read()
        # Reset file pointer in case it's needed again
        file.file.seek(0)
        
        if not data or len(data) == 0:
            error_msg = "Файл пуст или не может быть прочитан"
            logger.error(error_msg, extra={"trace_id": trace_id})
            return {"status": "error", "message": error_msg}
        
        
        content_type = file.content_type or ""
        filename = file.filename or "uploaded"
        
        text = ""
        doc_type = "file"
        
        try:
            if "pdf" in content_type or filename.lower().endswith(".pdf"):
                try:
                    text = extract_text_from_pdf_bytes(data)
                    doc_type = "pdf"
                except Exception as pdf_error:
                    error_msg = f"Ошибка при извлечении текста из PDF: {str(pdf_error)}"
                    logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
                    return {"status": "error", "message": error_msg}
            elif content_type.startswith("image/") or any(filename.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff")):
                try:
                    text = extract_text_from_image_bytes(data)
                    doc_type = "image"
                except Exception as img_error:
                    error_msg = f"Ошибка при OCR изображения: {str(img_error)}"
                    logger.error(error_msg, extra={"trace_id": trace_id}, exc_info=True)
                    return {"status": "error", "message": error_msg}
            else:
                try:
                    text = data.decode("utf-8")
                except Exception:
                    text = ""
                doc_type = "file"
        except Exception as e:
            logger.error(f"Error processing file: {e}", extra={"trace_id": trace_id}, exc_info=True)
            return {"status": "error", "message": f"Ошибка при обработке файла: {str(e)}"}
        
        if not text or not text.strip():
            return {"status": "error", "message": "Не удалось извлечь текст из файла. Возможно, файл пуст, зашифрован или содержит только изображения."}
        
        # Store using tool
        result = store_document.invoke({
            "source": filename,
            "content": text,
            "doc_type": doc_type
        })
        
        logger.info(f"File ingested successfully: {filename}", extra={"trace_id": trace_id})
        return {"status": "ok", "source": filename, "message": result}
    except Exception as e:
        logger.error(f"Ingest file error: {e}", extra={"trace_id": trace_id}, exc_info=True)
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


@app.post("/query")
def query(q: Question, request: Request):
    """Query using coordinator agent to determine intent and route to appropriate agent."""
    trace_id = getattr(request.state, "trace_id", trace_context.get_trace_id())
    trace_context.set_trace_id(trace_id)
    
    try:
        # Determine user intent using coordinator agent
        intent_result = determine_user_intent(q.question, trace_id)
        intent = intent_result.get("intent", "SEARCH")
        
        # Route to appropriate handler based on intent
        if intent == "SAVE_URL":
            url = intent_result.get("url")
            if url:
                # Use direct tool calls (more reliable than agent)
                try:
                    tool_start = time.time()
                    content = fetch_url_content.invoke({"url": url})
                    track_tool_call("fetch_url_content", success=not (content.startswith("Error") or content.startswith("Invalid")), duration=time.time() - tool_start)
                    
                    if content.startswith("Error") or content.startswith("Invalid"):
                        return {"answer": f"Ошибка при загрузке страницы: {content[:200]}"}
                    
                    if not content or len(content.strip()) < 50:
                        return {"answer": "Не удалось извлечь содержимое со страницы."}
                    
                    tool_start = time.time()
                    result = store_document.invoke({
                        "source": url,
                        "content": content,
                        "doc_type": "webpage"
                    })
                    track_tool_call("store_document", success=not result.startswith("Error"), duration=time.time() - tool_start)
                    
                    if result.startswith("Error"):
                        return {"answer": f"Ошибка при сохранении: {result[:200]}"}
                    
                    # Filter output for user-friendly message
                    is_safe, filtered_output = filter_output_content(result)
                    if "successfully" in filtered_output.lower() or "stored" in filtered_output.lower():
                        return {"answer": "Страница успешно сохранена в базу знаний."}
                    else:
                        return {"answer": filtered_output if is_safe else "Страница сохранена."}
                except Exception as e:
                    logger.error(f"Error saving URL: {e}", extra={"trace_id": trace_id}, exc_info=True)
                    return {"answer": f"Ошибка при сохранении страницы: {str(e)[:200]}"}
            else:
                return {"answer": "Не удалось извлечь URL из команды. Пожалуйста, укажите URL явно."}
        
        elif intent == "SEARCH":
            # Check if database is empty
            with driver.session() as session:
                check_empty_query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
                RETURN count(c) AS chunk_count
                LIMIT 1
                """
                result = session.run(check_empty_query)
                record = result.single()
                chunk_count = record["chunk_count"] if record else 0
                
                if chunk_count == 0:
                    return {
                        "answer": "База данных пуста. Пожалуйста, отправьте ссылку или файл, чтобы добавить данные в базу знаний, а затем задайте вопрос."
                    }
            
            # Use search agent for querying (following multi-agent architecture)
            query_text = intent_result.get("query", q.question)
            # Remove URL from query text if present (for cleaner search)
            url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
            query_text = url_pattern.sub('', query_text).strip()
            if not query_text:
                query_text = q.question
            
            try:
                # Try search agent first (following multi-agent architecture)
                answer = None
                agent_start_time = time.time()
                try:
                    callback = ObservabilityCallbackHandler(trace_id)
                    agent_result = search_agent.invoke(
                        {"input": query_text},
                        config={"callbacks": [callback]} if callback else {}
                    )
                    agent_duration = time.time() - agent_start_time
                    answer = agent_result.get("output", "") if isinstance(agent_result, dict) else str(agent_result)
                    
                    if answer and not answer.strip().startswith("Error") and len(answer.strip()) > 10:
                        # Agent succeeded
                        track_agent_invocation("search_agent", success=True, duration=agent_duration)
                        logger.info(f"Query processed successfully by search_agent", extra={"trace_id": trace_id})
                    else:
                        # Agent returned error or empty - use fallback
                        track_agent_invocation("search_agent", success=False, duration=agent_duration)
                        logger.warning(f"Search agent returned error or empty output, using direct tool calls", extra={"trace_id": trace_id})
                        answer = None
                except Exception as agent_error:
                    # Agent raised exception - use fallback
                    agent_duration = time.time() - agent_start_time
                    track_agent_invocation("search_agent", success=False, duration=agent_duration)
                    logger.warning(f"Search agent error: {type(agent_error).__name__}, using direct tool calls", extra={"trace_id": trace_id})
                    answer = None
                
                # Fallback to direct tool calls if agent failed
                if not answer:
                    logger.debug(f"Processing query using direct tool calls (fallback)", extra={"trace_id": trace_id})
                    tool_start = time.time()
                    context = search_graph_rag.invoke({"question": query_text, "limit": 20})
                    track_tool_call("search_graph_rag", success=not (context.startswith("Error") if context else False), duration=time.time() - tool_start)
                    if context and not context.startswith("Error") and len(context.strip()) > 10:
                        # Determine answer detail level
                        question_lower = query_text.lower()
                        detail_keywords = ["детально", "подробно", "подробный", "развернуто", "развернутый", "полный", "полностью", 
                                         "подробнее", "детальнее", "все", "все детали", "подробное описание"]
                        is_detailed = any(keyword in question_lower for keyword in detail_keywords)
                        
                        # For questions about "this article" or "эта статья", provide comprehensive answer
                        if "эта статья" in question_lower or "this article" in question_lower or "о чем" in question_lower:
                            is_detailed = True
                        
                        if is_detailed:
                            system_prompt = """Answer the question based ONLY on the provided context. Each chunk is labeled with [Source: ...] and [Type: ...] to indicate which document it comes from.

Your task:
1. Understand the semantic meaning of the question - what information is being asked for
2. Identify the target source from question through semantic understanding (match sources intelligently by understanding names, domains, descriptions)
3. Understand temporal context if mentioned in question (interpret relative time references semantically)
4. Use ONLY chunks from the relevant source/document that semantically matches the question context
5. If question mentions both source AND temporal context, use chunks that match BOTH criteria
6. DO NOT mix information from different sources - if question refers to a specific source, use ONLY chunks from that source
7. If chunks are from different sources, prioritize chunks from the source that semantically matches the question
8. Ignore chunks from irrelevant sources even if they contain similar words

Provide a clear, concise, and informative answer. Focus on the most important information. Use 2-4 paragraphs maximum. Be thorough but brief. Use Russian if question is in Russian, else English. Use plain text only - no markdown formatting."""
                        else:
                            system_prompt = """Answer the question based ONLY on the provided context. Each chunk is labeled with [Source: ...] and [Type: ...] to indicate which document it comes from.

Your task:
1. Understand the semantic meaning of the question - what information is being asked for
2. Identify the target source from question through semantic understanding (match sources intelligently by understanding names, domains, descriptions)
3. Understand temporal context if mentioned in question (interpret relative time references semantically)
4. Use ONLY chunks from the relevant source/document that semantically matches the question context
5. If question mentions both source AND temporal context, use chunks that match BOTH criteria
6. DO NOT mix information from different sources - if question refers to a specific source, use ONLY chunks from that source
7. If chunks are from different sources, prioritize chunks from the source that semantically matches the question
8. Ignore chunks from irrelevant sources even if they contain similar words

Provide a clear and concise answer. Focus on key information. Use 1-2 paragraphs maximum. Be brief and to the point. Use Russian if question is in Russian, else English. Use plain text only - no markdown formatting."""
                        
                        # Use reasonable context limit for concise answers
                        context_limit = 6000 if is_detailed else 4000
                        messages = [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=f"Question: {query_text}\n\nContext:\n{context[:context_limit]}")
                        ]
                        # Create callback for LLM metrics
                        callback = ObservabilityCallbackHandler(trace_id)
                        llm_start = time.time()
                        response = llm.invoke(messages, config={"callbacks": [callback]})
                        llm_duration = time.time() - llm_start
                        answer = response.content if hasattr(response, 'content') else str(response)
                        
                        # Track LLM call manually with tokens from response_metadata
                        tokens_dict = None
                        if hasattr(response, 'response_metadata') and response.response_metadata:
                            usage = response.response_metadata.get('token_usage', {})
                            if usage:
                                tokens_dict = {
                                    "prompt_tokens": usage.get("prompt_tokens", 0),
                                    "completion_tokens": usage.get("completion_tokens", 0),
                                    "total_tokens": usage.get("total_tokens", 0)
                                }
                        # Always track LLM call
                        track_llm_call(model=LLM_MODEL, success=True, duration=llm_duration, tokens=tokens_dict)
                    else:
                        answer = context if context else "No information found in knowledge base."
                
                # Filter output
                is_safe, filtered_answer = filter_output_content(answer)
                
                if not filtered_answer or len(filtered_answer.strip()) < 10:
                    lang = "ru" if any(c in query_text.lower() for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") else "en"
                    if lang == "ru":
                        return {"answer": "Информация в базе знаний не найдена."}
                    else:
                        return {"answer": "No information found in knowledge base."}
                
                return {"answer": filtered_answer.strip()}
            except Exception as e:
                logger.error(f"Search error: {e}", extra={"trace_id": trace_id}, exc_info=True)
                raise
        
        else:  # UNKNOWN
            return {"answer": "Не понял ваш запрос. Вы можете отправить ссылку для сохранения или задать вопрос по сохраненным данным."}
            
    except Exception as e:
        logger.error(f"Query error: {e}", extra={"trace_id": trace_id}, exc_info=True)
        lang = "ru" if any(c in q.question.lower() for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя") else "en"
        error_msg = f"Ошибка при обработке запроса: {str(e)[:200]}" if lang == "ru" else f"Error processing query: {str(e)[:200]}"
        return {"status": "error", "answer": error_msg}


@app.post("/query_time")
def query_time(tq: TimeQuery, request: Request):
    """Query with time filter."""
    trace_id = getattr(request.state, "trace_id", trace_context.get_trace_id())
    trace_context.set_trace_id(trace_id)
    
    try:
        logger.info(f"Processing time query: {tq.question[:100]}...", extra={"trace_id": trace_id})
        
        # Add time filter to question
        question_with_time = tq.question
        if tq.since:
            question_with_time += f" (since {tq.since})"
        if tq.until:
            question_with_time += f" (until {tq.until})"
        
        # Use search agent
        try:
            if hasattr(search_agent, 'invoke'):
                callback = ObservabilityCallbackHandler(trace_id)
                result = search_agent.invoke(
                    {"input": question_with_time},
                    config={"callbacks": [callback]} if callback else {}
                )
                answer = result.get("output", "") if isinstance(result, dict) else str(result)
            else:
                result = search_agent.invoke({"input": question_with_time})
                if hasattr(result, 'content'):
                    answer = result.content
                else:
                    answer = str(result)
        except Exception as agent_error:
            context = search_graph_rag.invoke({"question": question_with_time, "limit": 20})
            if context and not context.startswith("Error"):
                from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
                # Determine answer detail level based on question
                question_lower = question_with_time.lower()
                detail_keywords = ["детально", "подробно", "подробный", "развернуто", "развернутый", "полный", "полностью", 
                                 "подробнее", "детальнее", "все", "все детали", "подробное описание"]
                is_detailed = any(keyword in question_lower for keyword in detail_keywords)
                
                if is_detailed:
                    system_prompt = "Answer the question based on the provided context. Provide a comprehensive, detailed, and very thorough answer. Use ALL relevant information from the context to give a complete and exhaustive response. Structure your answer logically with multiple paragraphs if needed. Be extremely thorough and cover all important aspects, details, and nuances mentioned in the context. Use Russian if question is in Russian, else English. Use plain text only - do not use markdown formatting like **, *, __, #, `, or ```."
                else:
                    system_prompt = "Answer the question based on the provided context. Provide a clear, informative answer of moderate length (2-4 sentences or a short paragraph). Cover the main points and key information, but be concise. Use Russian if question is in Russian, else English. Use plain text only - do not use markdown formatting like **, *, __, #, `, or ```."
                
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Question: {question_with_time}\n\nContext:\n{context[:4000]}")
                ]
                # Create callback for LLM metrics
                callback = ObservabilityCallbackHandler(trace_id)
                llm_start = time.time()
                response = llm.invoke(messages, config={"callbacks": [callback]})
                llm_duration = time.time() - llm_start
                
                # Track LLM call manually with tokens from response_metadata
                tokens_dict = None
                if hasattr(response, 'response_metadata') and response.response_metadata:
                    usage = response.response_metadata.get('token_usage', {})
                    if usage:
                        tokens_dict = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                # Always track LLM call
                track_llm_call(model=LLM_MODEL, success=True, duration=llm_duration, tokens=tokens_dict)
                answer = response.content if hasattr(response, 'content') else str(response)
            else:
                answer = context
        
        # Filter output
        is_safe, filtered_answer = filter_output_content(answer)
        
        if not filtered_answer or len(filtered_answer.strip()) < 10:
            return {"answer": "No information found in the specified time range."}
        
        logger.info(f"Time query processed successfully", extra={"trace_id": trace_id})
        return {"answer": filtered_answer.strip()}
    except Exception as e:
        logger.error(f"Query_time error: {e}", extra={"trace_id": trace_id}, exc_info=True)
        return {"status": "error", "answer": f"Ошибка при обработке запроса: {str(e)[:200]}"}


@app.post("/clear")
def clear_all(request: Request):
    """Clear all data from the graph."""
    trace_id = getattr(request.state, "trace_id", trace_context.get_trace_id())
    trace_context.set_trace_id(trace_id)
    
    try:
        with driver.session() as session:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
            deleted = result.single()["deleted"]
            logger.info(f"Cleared {deleted} nodes", extra={"trace_id": trace_id})
            return {"status": "ok", "deleted": deleted}
    except Exception as e:
        logger.error(f"Clear error: {e}", extra={"trace_id": trace_id}, exc_info=True)
        return {"status": "error", "message": str(e)}


@app.get("/health")
def health():
    """Health check endpoint."""
    neo4j_ok = False
    documents_count = 0
    chunks_count = 0
    
    try:
        with driver.session() as session:
            session.run("RETURN 1")
            neo4j_ok = True
            
            # Get document and chunk counts
            doc_result = session.run("MATCH (d:Document) RETURN count(d) AS count")
            documents_count = doc_result.single()["count"] if doc_result.peek() else 0
            
            chunk_result = session.run("MATCH (c:Chunk) RETURN count(c) AS count")
            chunks_count = chunk_result.single()["count"] if chunk_result.peek() else 0
            
            # Update metrics
            update_neo4j_counts(documents_count, chunks_count)
    except Exception as e:
        logger.debug(f"Neo4j health check error: {e}")
    
    llm_ok = OPENROUTER_API_KEY is not None
    
    # Test DNS resolution
    dns_ok = False
    try:
        socket.gethostbyname("www.php.net")
        dns_ok = True
    except Exception:
        pass
    
    
    return {
        "status": "ok" if neo4j_ok else "error",
        "neo4j": neo4j_ok,
        "llm_configured": llm_ok,
        "dns_resolution": dns_ok,
        "observability": True,
        "guardrails": ENABLE_INPUT_FILTERING and ENABLE_OUTPUT_FILTERING
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response
    return Response(content=get_metrics(), media_type=get_metrics_content_type())



