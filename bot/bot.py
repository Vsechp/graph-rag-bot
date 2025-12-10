import os
import re
import asyncio
import logging
import uuid
import time
from threading import Thread
from http.server import HTTPServer, BaseHTTPRequestHandler
from dotenv import load_dotenv
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command
from aiogram.types import BotCommand

# Metrics using prometheus_client directly
try:
    from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
    
    # Telegram metrics
    telegram_messages_total = Counter('telegram_messages_total', 'Total number of Telegram messages received', ['message_type'])
    telegram_errors_total = Counter('telegram_errors_total', 'Total number of Telegram bot errors', ['error_type'])
    telegram_response_duration_seconds = Histogram(
        'telegram_response_duration_seconds', 
        'Telegram bot response duration in seconds', 
        ['handler_type'],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    )
    
    def track_telegram_message(message_type: str):
        telegram_messages_total.labels(message_type=message_type).inc()
    
    def track_telegram_error(error_type: str):
        telegram_errors_total.labels(error_type=error_type).inc()
    
    def track_telegram_response(handler_type: str, duration: float):
        telegram_response_duration_seconds.labels(handler_type=handler_type).observe(duration)
    
    def get_metrics():
        return generate_latest()
    
    def get_metrics_content_type():
        return CONTENT_TYPE_LATEST
    
    METRICS_AVAILABLE = True
    logging.info("Metrics initialized successfully")
except ImportError as e:
    logging.warning(f"Metrics not available: {e}")
    METRICS_AVAILABLE = False
    def track_telegram_message(*args, **kwargs): pass
    def track_telegram_error(*args, **kwargs): pass
    def track_telegram_response(*args, **kwargs): pass
    def get_metrics(): return b"# Metrics not available\n"
    def get_metrics_content_type(): return "text/plain"

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("TELEGRAM_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL")

bot = Bot(token=TOKEN)
dp = Dispatcher()

# Set bot commands menu
async def set_bot_commands():
    commands = [
        BotCommand(command="start", description="Начать работу с ботом"),
        BotCommand(command="help", description="Помощь и инструкции"),
        BotCommand(command="clear", description="Очистить всю базу данных"),
    ]
    await bot.set_my_commands(commands)

URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?:/.*)?")


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
    
    # Clean up extra spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines
    text = re.sub(r' {2,}', ' ', text)  # Multiple spaces
    
    return text.strip()


@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.reply(
        "Привет! Я AI-помощник с GraphRAG.\n\n"
        "Отправьте мне ссылку на сайт, документ или картинку - я скачаю данные и сохраню в RAG.\n"
        "После этого вы сможете задавать вопросы по сохраненным данным.\n\n"
        "Используйте /help для получения справки."
    )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "📚 <b>ГрафRAG Бот - Справка</b>\n\n"
        "🔹 <b>Как использовать:</b>\n"
        "• Отправьте ссылку на сайт для индексации\n"
        "• Отправьте PDF файл или изображение для обработки\n"
        "• Задайте вопрос по сохраненным данным\n\n"
        "🔹 <b>Команды:</b>\n"
        "/start - Начать работу\n"
        "/help - Показать эту справку\n"
        "/clear - Очистить всю базу данных\n\n"
        "🔹 <b>Примеры вопросов:</b>\n"
        "• О чем этот сайт?\n"
        "• Что есть на этом сайте?\n"
        "• Расскажи подробнее о..."
    )
    await message.reply(help_text, parse_mode="HTML")


def get_trace_id() -> str:
    """Generate trace ID for distributed tracing."""
    return str(uuid.uuid4())


@dp.message(Command("clear"))
async def cmd_clear(message: types.Message):
    trace_id = get_trace_id()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/clear",
                headers={"X-Trace-Id": trace_id, "Trace-Id": trace_id}
            )
            if resp.status_code == 200:
                data = resp.json()
                deleted = data.get("deleted", 0)
                await message.reply(f"База данных очищена. Удалено узлов: {deleted}")
            else:
                await message.reply(f"Ошибка: HTTP {resp.status_code}")
    except Exception as e:
        logging.error(f"Error clearing DB: {e}")
        await message.reply(f"Ошибка при очистке: {e}")


def is_url(text: str) -> bool:
    text = text.strip()
    if not text:
        return False
    
    if text.startswith("http://") or text.startswith("https://"):
        return True
    
    if " " in text:
        return False
    
    if "." not in text:
        return False
    
    parts = text.split(".")
    if len(parts) < 2:
        return False
    
    if not all(len(p) > 0 for p in parts):
        return False
    
    last_part = parts[-1]
    if len(last_part) < 2:
        return False
    
    if any(c in text for c in ["?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]):
        return False
    
    return True


def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


@dp.message()
async def handle_message(message: types.Message):
    text = message.text or ""
    
    if text.startswith('/'):
        return
    
    if message.document:
        await message.reply("Скачиваю файл и сохраняю в RAG...")
        await upload_file_to_backend(message, message.document)
        return

    if message.photo:
        await message.reply("Обрабатываю картинку и сохраняю в RAG...")
        photo = message.photo[-1]
        await upload_file_to_backend(message, photo)
        return

    if is_url(text):
        url = normalize_url(text)
        status_msg = await message.reply("Скачиваю данные с сайта и сохраняю в RAG. Это может занять до минуты...")
        await send_url_to_backend(message, url, status_msg)
        return

    if text.strip():
        status_msg = await message.reply("Ищу информацию в RAG...")
        await ask_backend_question(message, text, status_msg)
        return

    await message.reply(
        "Отправьте ссылку на сайт, документ или картинку для сохранения в RAG,\n"
        "или задайте вопрос по сохраненным данным."
    )


async def upload_file_to_backend(message: types.Message, file_obj):
    try:
        file_info = await bot.get_file(file_obj.file_id)
        file_path = file_info.file_path
        file_bytes = await bot.download_file(file_path)
        content = file_bytes.read()
        
        # Determine filename and content type
        filename = getattr(file_obj, 'file_name', None) or file_path.split('/')[-1] if file_path else "uploaded"
        
        # Determine content type based on file extension
        content_type = None
        if filename.lower().endswith('.pdf'):
            content_type = 'application/pdf'
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            content_type = f'image/{filename.split(".")[-1].lower()}'
        
        trace_id = get_trace_id()
        headers = {"X-Trace-Id": trace_id, "Trace-Id": trace_id}
        async with httpx.AsyncClient(timeout=60.0) as client:
            files = {"file": (filename, content, content_type) if content_type else (filename, content)}
            resp = await client.post(
                f"{BACKEND_URL}/ingest/file",
                files=files,
                headers=headers
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    await message.reply(f"Данные сохранены в RAG.\nИсточник: {data.get('source')}\n\nТеперь вы можете задавать вопросы по этим данным.")
                else:
                    await message.reply(f"Ошибка при сохранении: {data.get('message', str(data))}")
            else:
                await message.reply(f"Ошибка от бэкенда: HTTP {resp.status_code}")
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        await message.reply(f"Ошибка при загрузке файла: {e}")


async def send_url_to_backend(message: types.Message, url: str, status_msg=None):
    trace_id = get_trace_id()
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/ingest/url",
                json={"url": url},
                headers={"X-Trace-Id": trace_id, "Trace-Id": trace_id}
            )
            
            duration = time.time() - start_time
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    if METRICS_AVAILABLE:
                        track_telegram_response("url_ingest", duration)
                    if status_msg:
                        await status_msg.edit_text(f"✅ Данные сохранены в RAG.\nИсточник: {data.get('source')}\n\nТеперь вы можете задавать вопросы по этим данным.")
                    else:
                        await message.reply(f"✅ Данные сохранены в RAG.\nИсточник: {data.get('source')}\n\nТеперь вы можете задавать вопросы по этим данным.")
                else:
                    if METRICS_AVAILABLE:
                        track_telegram_response("url_ingest", duration)
                    if status_msg:
                        await status_msg.edit_text(f"❌ Ошибка при сохранении: {data.get('message', str(data))}")
                    else:
                        await message.reply(f"❌ Ошибка при сохранении: {data.get('message', str(data))}")
            else:
                if METRICS_AVAILABLE:
                    track_telegram_response("url_ingest", duration)
                if status_msg:
                    await status_msg.edit_text(f"❌ Ошибка от бэкенда: HTTP {resp.status_code}")
                else:
                    await message.reply(f"❌ Ошибка от бэкенда: HTTP {resp.status_code}")
    except httpx.TimeoutException:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            track_telegram_response("url_ingest", duration)
        error_msg = "Превышено время ожидания. Попробуйте позже или используйте другую ссылку."
        logging.error(f"Timeout sending URL: {url}")
        if status_msg:
            await status_msg.edit_text(f"❌ Ошибка при обработке ссылки: {error_msg}")
        else:
            await message.reply(f"❌ Ошибка при обработке ссылки: {error_msg}")
    except Exception as e:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            track_telegram_response("url_ingest", duration)
        error_msg = str(e) if str(e) else "Неизвестная ошибка"
        logging.error(f"Error sending URL: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Ошибка при обработке ссылки: {error_msg}")
        else:
            await message.reply(f"❌ Ошибка при обработке ссылки: {error_msg}")


async def ask_backend_question(message: types.Message, question: str, status_msg=None):
    start_time = time.time()
    since = None
    until = None
    
    since_match = re.search(r'since:(\d{4}-\d{2}-\d{2})', question)
    until_match = re.search(r'until:(\d{4}-\d{2}-\d{2})', question)
    
    if since_match:
        since = since_match.group(1)
        question = re.sub(r'since:\d{4}-\d{2}-\d{2}\s*', '', question).strip()
    
    if until_match:
        until = until_match.group(1)
        question = re.sub(r'until:\d{4}-\d{2}-\d{2}\s*', '', question).strip()
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if since or until:
                payload = {"question": question}
                if since:
                    payload["since"] = since
                if until:
                    payload["until"] = until
                resp = await client.post(f"{BACKEND_URL}/query_time", json=payload)
            else:
                resp = await client.post(f"{BACKEND_URL}/query", json={"question": question})
            
            duration = time.time() - start_time
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("answer")
                if METRICS_AVAILABLE:
                    track_telegram_response("query", duration)
                if not answer:
                    if data.get("context"):
                        context_preview = "\n\n---\n\n".join(data.get("context")[:3])
                        if status_msg:
                            await status_msg.edit_text(f"Найденные фрагменты:\n\n{context_preview[:1500]}")
                        else:
                            await message.reply(f"Найденные фрагменты:\n\n{context_preview[:1500]}")
                    else:
                        if status_msg:
                            await status_msg.edit_text("Ответ не получен.")
                        else:
                            await message.reply("Ответ не получен.")
                else:
                    # Clean markdown from answer
                    clean_answer = clean_markdown(answer)
                    if status_msg:
                        await status_msg.edit_text(clean_answer[:4000])
                    else:
                        await message.reply(clean_answer[:4000])
            else:
                if METRICS_AVAILABLE:
                    track_telegram_response("query", duration)
                error_text = ""
                try:
                    error_data = resp.json()
                    error_text = error_data.get("message") or error_data.get("answer") or str(error_data)
                except:
                    error_text = resp.text[:500] if resp.text else "Неизвестная ошибка"
                if status_msg:
                    await status_msg.edit_text(f"❌ Ошибка от бэкенда (HTTP {resp.status_code}): {error_text}")
                else:
                    await message.reply(f"❌ Ошибка от бэкенда (HTTP {resp.status_code}): {error_text}")
    except httpx.TimeoutException:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            track_telegram_response("query", duration)
        if status_msg:
            await status_msg.edit_text("⏱ Превышено время ожидания ответа от бэкенда. Попробуйте еще раз.")
        else:
            await message.reply("⏱ Превышено время ожидания ответа от бэкенда. Попробуйте еще раз.")
    except httpx.RequestError as e:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            track_telegram_response("query", duration)
        logging.error(f"Request error: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Ошибка подключения к бэкенду: {str(e)[:200]}")
        else:
            await message.reply(f"❌ Ошибка подключения к бэкенду: {str(e)[:200]}")
    except Exception as e:
        duration = time.time() - start_time
        if METRICS_AVAILABLE:
            track_telegram_response("query", duration)
        logging.error(f"Error asking question: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Ошибка при запросе к бэкенду: {str(e)[:200]}")
        else:
            await message.reply(f"❌ Ошибка при запросе к бэкенду: {str(e)[:200]}")


async def main():
    """Main function to start the bot."""
    # Set bot commands menu
    await set_bot_commands()
    print("Bot commands menu set successfully")
    
    # Start polling
    print("Starting bot...")
    await dp.start_polling(bot)


# Metrics HTTP server
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', get_metrics_content_type())
            self.end_headers()
            self.wfile.write(get_metrics())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs


def start_metrics_server():
    """Start HTTP server for metrics on port 9091"""
    server = HTTPServer(('0.0.0.0', 9091), MetricsHandler)
    server.serve_forever()


if __name__ == "__main__":
    # Start metrics server in background thread
    if METRICS_AVAILABLE:
        metrics_thread = Thread(target=start_metrics_server, daemon=True)
        metrics_thread.start()
        logging.info("Metrics server started on port 9091")
    
    try:
        asyncio.run(main())
    finally:
        asyncio.run(bot.session.close())
