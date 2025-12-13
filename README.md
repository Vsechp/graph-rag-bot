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
- **LangChain** - Многоагентная система (координатор, url-агент, search-агент)
- **OpenRouter** - LLM сервис для генерации ответов (модель `qwen/qwen3-30b-a3b-thinking-2507`)
- **FastAPI** - REST API backend
- **Aiogram** - Telegram бот фреймворк

## Многоагентная система

Проект использует многоагентную архитектуру на LangChain:

1. **Координатор агент** - определяет намерение пользователя (SAVE_URL или SEARCH) через LLM и маршрутизирует запросы к соответствующим агентам
2. **URL агент** - обрабатывает сохранение веб-страниц (загрузка контента и сохранение в Neo4j)
3. **Search агент** - отвечает на вопросы через графовую RAG базу данных

Новых агентов можно добавлять через `create_agent_executor()` с нужными tools и промптом, затем добавить маршрутизацию в координатор.

## API Endpoints

- `POST /ingest/url` - Сохранение URL в базу знаний
- `POST /ingest/file` - Сохранение файла (PDF, изображение)
- `POST /query` - Поиск и ответ на вопрос по базе знаний
- `POST /query_time` - Поиск с фильтром по времени
- `POST /clear` - Очистка всей базы данных
- `GET /health` - Проверка здоровья сервиса
- `GET /metrics` - Prometheus метрики для мониторинга


## Настройка Grafana для просмотра метрик

### 1. Запустить Prometheus и Grafana
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

### 2. Открыть Grafana
- Откройте http://localhost:3000
- **Логин:** `admin`
- **Пароль:** `admin`

### 3. Добавить Prometheus как источник данных
1. **Configuration** → **Data Sources** → **Add data source**
2. Выберите **Prometheus**
3. **URL:** `http://prometheus:9090`
4. Нажмите **Save & Test**

### 4. Импортировать готовый дашборд
1. Нажмите **+** (в левом меню) → **Import**
2. Нажмите **Upload JSON file**
3. Выберите файл `grafana-dashboard.json` из корня проекта
4. Выберите **Prometheus** как источник данных
5. Нажмите **Import**

Готово! Дашборд с метриками загружен.

**Что показывает дашборд:**
- Вызовы агентов и их производительность
- Вызовы LLM и использование токенов
- Вызовы инструментов
- HTTP запросы и время ответа
- Количество документов и chunks в Neo4j
- Telegram сообщения
- Guardrails блокировки

<img width="3388" height="1698" alt="Снимок экрана 2025-12-13 в 21 32 11" src="https://github.com/user-attachments/assets/bcb5c89f-c8a9-40bb-ad87-70cd5d8ede11" />
Метрики в Graphana
**Примечание:** Некоторые метрики могут быть пустыми, если они еще не использовались (например, метрики Telegram ошибок появятся только при возникновении ошибок).

## Экспорт данных из Grafana

### Экспорт дашборда
1. Откройте дашборд
2. Нажмите на иконку **⚙️ Settings** (вверху справа)
3. Нажмите **JSON Model** - скопируйте JSON
4. Или нажмите **Share** → **Export** → **Save to file**

### Экспорт данных панели
1. Откройте панель (нажмите на заголовок панели)
2. Нажмите **...** (три точки) → **Inspect**
3. Выберите **Data** - увидите сырые данные
4. Нажмите **Download CSV** для экспорта

## Просмотр данных в Neo4j

### Через Neo4j Browser (веб-интерфейс)
1. Откройте http://localhost:7474
2. **Логин:** `neo4j`
3. **Пароль:** из `.env` файла (`NEO4J_PASSWORD`)
4. Выполните Cypher запросы в консоли

### Через командную строку (cypher-shell)
```bash
# Подключиться к Neo4j
docker-compose exec neo4j cypher-shell -u neo4j -p <PASSWORD>

# Или если пароль в .env
docker-compose exec neo4j cypher-shell -u neo4j -p $(grep NEO4J_PASSWORD .env | cut -d '=' -f2)
```

### Cypher запросы

**Посмотреть все документы:**
```cypher
MATCH (d:Document)
RETURN d.source, d.type, d.created_at
ORDER BY d.created_at DESC
LIMIT 10;
```

**Посмотреть количество документов и chunks:**
```cypher
MATCH (d:Document)
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
RETURN count(DISTINCT d) AS documents, count(c) AS chunks;
```

**Посмотреть последний документ и его chunks:**
```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)
WHERE d.created_at = (SELECT max(d2.created_at) FROM Document d2)
RETURN d.source, d.type, count(c) AS chunk_count, 
       collect(c.text)[0..3] AS sample_chunks;
```

**Посмотреть структуру графа (визуализация):**
```cypher
MATCH (d:Document)-[r:HAS_CHUNK]->(c:Chunk)
RETURN d, r, c
LIMIT 50;
```

**Поиск по тексту в chunks:**
```cypher
MATCH (c:Chunk)
WHERE c.text CONTAINS 'ваш_поисковый_запрос'
RETURN c.text, c.chunk_index
LIMIT 10;
```

**Статистика по типам документов:**
```cypher
MATCH (d:Document)
RETURN d.type, count(*) AS count
ORDER BY count DESC;
```

**Посмотреть все узлы:**
```cypher
MATCH (n)
RETURN labels(n) AS labels, count(*) AS count;
```

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
