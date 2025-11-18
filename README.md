# GraphRAG Bot

Telegram бот с GraphRAG для обработки ссылок и ответов на вопросы на основе графа знаний Neo4j.

## Быстрый старт

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Vsechp/graph-rag-bot.git
cd graph-rag-bot
```

2. Создайте `.env` файл из примера:
```bash
cp .env.example .env
```

3. Отредактируйте `.env` и укажите свои значения:
   - `TELEGRAM_TOKEN` - токен Telegram бота
   - `OPENROUTER_API_KEY` - API ключ OpenRouter
   - `NEO4J_PASSWORD` - пароль для Neo4j

4. Запустите через Docker Compose:
```bash
docker-compose up -d
```

5. Проверьте статус сервисов:
```bash
docker-compose ps
```

Сервисы будут доступны:
- Backend API: http://localhost:8000
- Neo4j Browser: http://localhost:7474
- Telegram бот: готов к использованию

## Использование

Отправьте боту в Telegram:
- Ссылку для сохранения: `https://example.com`
- PDF файл или изображение для обработки
- Вопрос для поиска: `о чем этот сайт?` или любой другой вопрос

## Архитектура

Проект состоит из трех сервисов:

- **neo4j** - Графовая база данных Neo4j (порт 7474, 7687)
- **backend** - FastAPI приложение с LangChain агентами (порт 8000)
- **bot** - Telegram бот на aiogram

Технологии:
- **Neo4j** - Графовая база данных для хранения знаний
- **LangChain** - Многоагентная система (координатор, search-агент, tools)
- **OpenRouter** - LLM сервис для генерации ответов
- **FastAPI** - REST API backend
- **Aiogram** - Telegram бот фреймворк

## API Endpoints

- `POST /ingest/url` - Сохранение URL в базу знаний
- `POST /ingest/file` - Сохранение файла (PDF, изображение)
- `POST /query` - Поиск и ответ на вопрос по базе знаний
- `POST /query_time` - Поиск с фильтром по времени
- `POST /clear` - Очистка всей базы данных
- `GET /health` - Проверка здоровья сервиса

## Управление Docker сервисами

```bash
# Запуск всех сервисов
docker-compose up -d

# Остановка всех сервисов
docker-compose down

# Просмотр логов
docker-compose logs -f

# Просмотр логов конкретного сервиса
docker-compose logs -f backend
docker-compose logs -f bot
docker-compose logs -f neo4j

# Перезапуск сервиса
docker-compose restart backend
docker-compose restart bot

# Полная очистка (удаление всех данных и volumes)
docker-compose down -v
docker volume prune -f

# Пересборка образов
docker-compose build
docker-compose up -d
```
