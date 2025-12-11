"""
Prometheus metrics для мониторинга Telegram бота и агентов.
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from typing import Optional

# ==================== HTTP метрики ====================

# Счетчики запросов
http_requests_total = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

# Время выполнения запросов
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    buckets=[
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    ],
)

# ==================== Telegram Bot метрики ====================

# Счетчики сообщений от пользователей
telegram_messages_total = Counter(
    "telegram_messages_total",
    "Total number of Telegram messages received",
    ["message_type"],  # text, url, file, image
)

# Ошибки бота
telegram_errors_total = Counter(
    "telegram_errors_total", "Total number of Telegram bot errors", ["error_type"]
)

# Время ответа бота
telegram_response_duration_seconds = Histogram(
    "telegram_response_duration_seconds",
    "Telegram bot response duration in seconds",
    ["handler_type"],  # url_ingest, query, file_ingest
)

# ==================== Agent метрики ====================

# Вызовы агентов
agent_invocations_total = Counter(
    "agent_invocations_total",
    "Total number of agent invocations",
    ["agent_name", "status"],  # coordinator, url_agent, search_agent, success/failure
)

# Время выполнения агентов
agent_duration_seconds = Histogram(
    "agent_duration_seconds",
    "Agent execution duration in seconds",
    ["agent_name"],
    buckets=[
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        0.75,
        1.0,
        2.5,
        5.0,
        7.5,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
        600.0,
    ],
)

# Количество итераций агента
agent_iterations = Histogram(
    "agent_iterations", "Number of agent iterations", ["agent_name"]
)

# ==================== LLM метрики ====================

# Вызовы LLM
llm_calls_total = Counter(
    "llm_calls_total",
    "Total number of LLM calls",
    ["model", "status"],  # success/failure
)

# Токены LLM
llm_tokens_total = Counter(
    "llm_tokens_total",
    "Total number of LLM tokens",
    ["model", "type"],  # prompt/completion/total
)

# Время ответа LLM
llm_duration_seconds = Histogram(
    "llm_duration_seconds", "LLM response duration in seconds", ["model"]
)

# ==================== Tool метрики ====================

# Вызовы инструментов
tool_calls_total = Counter(
    "tool_calls_total",
    "Total number of tool calls",
    [
        "tool_name",
        "status",
    ],  # fetch_url_content, store_document, search_graph_rag, success/failure
)

# Время выполнения инструментов
tool_duration_seconds = Histogram(
    "tool_duration_seconds", "Tool execution duration in seconds", ["tool_name"]
)

# ==================== Neo4j метрики ====================

# Количество документов в базе
neo4j_documents_count = Gauge(
    "neo4j_documents_count", "Number of documents in Neo4j database"
)

# Количество chunks в базе
neo4j_chunks_count = Gauge("neo4j_chunks_count", "Number of chunks in Neo4j database")

# Операции Neo4j
neo4j_operations_total = Counter(
    "neo4j_operations_total",
    "Total number of Neo4j operations",
    [
        "operation_type",
        "status",
    ],  # create_document, create_chunk, search, success/failure
)

# Время выполнения операций Neo4j
neo4j_duration_seconds = Histogram(
    "neo4j_duration_seconds", "Neo4j operation duration in seconds", ["operation_type"]
)

# ==================== Guardrails метрики ====================

# Блокировки guardrails
guardrails_blocks_total = Counter(
    "guardrails_blocks_total",
    "Total number of guardrail blocks",
    ["block_type"],  # prompt_injection, pii, invalid_url
)

# ==================== Функции для обновления метрик ====================


def track_http_request(method: str, endpoint: str, status: int, duration: float):
    """Отслеживание HTTP запроса."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
        duration
    )


def track_telegram_message(message_type: str):
    """Отслеживание сообщения от пользователя."""
    telegram_messages_total.labels(message_type=message_type).inc()


def track_telegram_error(error_type: str):
    """Отслеживание ошибки бота."""
    telegram_errors_total.labels(error_type=error_type).inc()


def track_telegram_response(handler_type: str, duration: float):
    """Отслеживание времени ответа бота."""
    telegram_response_duration_seconds.labels(handler_type=handler_type).observe(
        duration
    )


def track_agent_invocation(
    agent_name: str,
    success: bool,
    duration: Optional[float] = None,
    iterations: Optional[int] = None,
):
    """Отслеживание вызова агента."""
    status = "success" if success else "failure"
    agent_invocations_total.labels(agent_name=agent_name, status=status).inc()

    if duration is not None:
        agent_duration_seconds.labels(agent_name=agent_name).observe(duration)

    if iterations is not None:
        agent_iterations.labels(agent_name=agent_name).observe(iterations)


def track_llm_call(
    model: str, success: bool, duration: float, tokens: Optional[dict] = None
):
    """Отслеживание вызова LLM."""
    # Ensure model is not None
    if not model:
        model = "unknown"

    status = "success" if success else "failure"
    llm_calls_total.labels(model=model, status=status).inc()
    llm_duration_seconds.labels(model=model).observe(duration)

    if tokens:
        for token_type, count in tokens.items():
            if token_type in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                llm_tokens_total.labels(model=model, type=token_type).inc(count)


def track_tool_call(tool_name: str, success: bool, duration: Optional[float] = None):
    """Отслеживание вызова инструмента."""
    status = "success" if success else "failure"
    tool_calls_total.labels(tool_name=tool_name, status=status).inc()

    if duration is not None:
        tool_duration_seconds.labels(tool_name=tool_name).observe(duration)


def track_neo4j_operation(
    operation_type: str, success: bool, duration: Optional[float] = None
):
    """Отслеживание операции Neo4j."""
    status = "success" if success else "failure"
    neo4j_operations_total.labels(operation_type=operation_type, status=status).inc()

    if duration is not None:
        neo4j_duration_seconds.labels(operation_type=operation_type).observe(duration)


def update_neo4j_counts(documents: int, chunks: int):
    """Обновление счетчиков документов и chunks."""
    neo4j_documents_count.set(documents)
    neo4j_chunks_count.set(chunks)


def track_guardrail_block(block_type: str):
    """Отслеживание блокировки guardrail."""
    guardrails_blocks_total.labels(block_type=block_type).inc()


def get_metrics():
    """Получить метрики в формате Prometheus."""
    return generate_latest()


def get_metrics_content_type():
    """Получить Content-Type для метрик."""
    return CONTENT_TYPE_LATEST
