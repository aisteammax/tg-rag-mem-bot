# Техническое задание: исправления и улучшения `tg-rag-mem-bot`

**Репозиторий:** https://github.com/aisteammax/tg-rag-mem-bot  
**Основной файл:** `bot.py` (448 строк)  
**Стек:** Python 3.10+, Aiogram 3.x, Qdrant, SQLite, aiohttp

---

## Контекст

Бот представляет собой Telegram-ассистента с RAG-памятью на базе Qdrant.
Пайплайн: запрос пользователя → Query Rewriting (OpenRouter) → векторный поиск (Qdrant) → локальный реранкер (Cross-Encoder) → ответ LLM со стримингом.

Задачи ниже разделены на два блока: **P0** (критические баги, блокируют работу) и **P1** (улучшения качества кода). Каждая задача содержит точное описание проблемы, ожидаемое поведение и пример реализации.

---

## P0 — Критические баги (исправить обязательно)

### P0-1. Отсутствует `import time`

**Файл:** `bot.py`  
**Проблема:** В функции `chat_handler` используется `time.time()` для throttling стриминга ответа, но модуль `time` нигде не импортирован. При первом текстовом сообщении бот упадёт с `NameError: name 'time' is not defined`.

**Что сделать:** Добавить `import time` в блок стандартных импортов в начале файла (строки 1–10), рядом с `import asyncio`, `import os` и т.д.

**Ожидаемый результат:** Бот обрабатывает текстовые сообщения без `NameError`.

---

### P0-2. Отсутствует завершение процесса при невалидном токене

**Файл:** `bot.py`, ~строка 26  
**Проблема:** При отсутствии переменной окружения `TELEGRAM_BOT_TOKEN` код только логирует ошибку, но продолжает выполнение. `Bot(token=None)` создаётся без исключения, но падает позже с малоинформативной ошибкой в глубине `aiogram`.

**Текущий код:**
```python
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не задан в переменных окружения!")

bot = Bot(token=BOT_TOKEN)
```

**Что сделать:** После логирования добавить `sys.exit(1)`. Также добавить `import sys` в импорты.

**Ожидаемый результат:**
```python
import sys

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN не задан в переменных окружения!")
    sys.exit(1)

bot = Bot(token=BOT_TOKEN)
```
При запуске без токена процесс завершается сразу с понятным сообщением в логах.

---

## P1 — Улучшения качества кода

### P1-1. Переиспользование `aiohttp.ClientSession`

**Файл:** `bot.py`  
**Проблема:** Во всех функциях (`fetch_url_content`, `chat_handler`, `search_cmd`) создаётся новый `aiohttp.ClientSession()` на каждый запрос. Это медленно: создание сессии включает DNS-резолвинг, TCP handshake и выделение ресурсов. При высокой нагрузке это приводит к излишнему расходу соединений.

**Что сделать:**

1. Объявить глобальную переменную сессии:
```python
http_session: aiohttp.ClientSession | None = None
```

2. Инициализировать её в `main()` перед запуском поллинга:
```python
async def main():
    global http_session
    http_session = aiohttp.ClientSession()
    ...
    try:
        await dp.start_polling(bot)
    finally:
        await http_session.close()
        await bot.session.close()
```

3. Заменить все вхождения `async with aiohttp.ClientSession() as session:` на использование глобального `http_session`. Пример:
```python
# Было:
async with aiohttp.ClientSession() as session:
    async with session.get(jina_url, ...) as response:
        ...

# Стало:
async with http_session.get(jina_url, ...) as response:
    ...
```

**Файлы для проверки:** все `aiohttp.ClientSession()` в `bot.py` (минимум 4 вхождения).

**Ожидаемый результат:** Единственная сессия создаётся при старте и закрывается при завершении. Нет `async with aiohttp.ClientSession()` внутри функций-обработчиков.

---

### P1-2. Обработка ошибок фоновой задачи `asyncio.create_task`

**Файл:** `bot.py`, конец функции `chat_handler`  
**Проблема:** 
```python
asyncio.create_task(add_to_vector_db(user_id, memory_chunk))
```
Если задача завершается с исключением, Python выводит предупреждение в stderr, но исключение молча теряется. Долгосрочная память пользователя не сохраняется, и никто об этом не узнает.

**Что сделать:** Добавить callback для логирования ошибок:
```python
def _log_task_error(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error(
            "Ошибка фонового сохранения в векторную БД: %s",
            task.exception(),
            exc_info=task.exception(),
        )

task = asyncio.create_task(add_to_vector_db(user_id, memory_chunk))
task.add_done_callback(_log_task_error)
```

Функцию `_log_task_error` объявить на уровне модуля (не внутри хендлера).

**Ожидаемый результат:** Все ошибки сохранения в Qdrant попадают в лог с трейсбеком.

---

### P1-3. Вынести импорты из тела хендлера на уровень модуля

**Файл:** `bot.py`  
**Проблема:** В теле `chat_handler` есть три отложенных импорта, которые нарушают PEP 8 и скрывают зависимости — ошибки `ImportError` появятся только при обработке первого файла нужного типа, а не при старте:

```python
# Внутри chat_handler:
import pypdf
import docx
from db.vector_db import get_embedding
```

**Что сделать:** Перенести все три импорта в начало файла, в соответствующие блоки:
- `from db.vector_db import get_embedding` — в блок импортов из `db/`
- `import pypdf` и `import docx` — после стандартных импортов, перед `aiogram`-импортами (обернуть в `try/except ImportError` с понятным сообщением об ошибке, если пакет не установлен):

```python
try:
    import pypdf
except ImportError:
    pypdf = None
    logger.warning("pypdf не установлен. PDF без Jina API обрабатываться не будут.")

try:
    import docx
except ImportError:
    docx = None
    logger.warning("python-docx не установлен. DOCX-файлы обрабатываться не будут.")
```

В хендлере при использовании добавить проверку:
```python
if pypdf is None:
    await status_msg.edit_text("❌ Локальный парсер PDF недоступен. Установите pypdf.")
    return
```

**Ожидаемый результат:** Все зависимости видны в начале файла. Отсутствие пакета обнаруживается при старте, а не при первом запросе.

---

### P1-4. Вынести повторяющуюся логику чанкинга в отдельную функцию

**Файл:** `bot.py`  
**Проблема:** Код создания чанков дублируется три раза (при обработке файлов, URL и результатов поиска) с одними и теми же магическими числами `chunk_size=1500`, `overlap=300`. При изменении параметров нужно менять в трёх местах.

**Пример одного из дублирующихся блоков:**
```python
chunk_size = 1500
overlap = 300
chunks = [f"Фрагмент из файла '{file_name}':\n{text[i:i + chunk_size]}"
          for i in range(0, len(text), chunk_size - overlap)]
```

**Что сделать:** Создать функцию-утилиту на уровне модуля:
```python
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300

def make_chunks(text: str, source_label: str) -> list[str]:
    """Разбивает текст на перекрывающиеся чанки для индексации в векторной БД."""
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(text), step):
        chunk = text[i : i + CHUNK_SIZE]
        chunks.append(f"{source_label}:\n{chunk}")
    return chunks
```

Заменить все три inline-блока вызовами этой функции. Примеры:
```python
# Файл:
chunks = make_chunks(file_content, f"Фрагмент из файла '{file_name}'")

# URL:
chunks = make_chunks(page_text, f"Статья по ссылке ({url})")

# Поиск:
chunks = make_chunks(text, f"Результаты веб-поиска по запросу '{query}'")
```

**Ожидаемый результат:** Ноль дублирующихся блоков чанкинга. Параметры `CHUNK_SIZE` и `CHUNK_OVERLAP` — константы уровня модуля.

---

### P1-5. Добавить `/search` в меню команд Telegram

**Файл:** `bot.py`, функция `main()`  
**Проблема:** Команда `/search` есть в тексте `/start` и имеет свой хендлер, но отсутствует в списке `commands`. Пользователи не видят её в меню Telegram.

**Текущий код:**
```python
commands = [
    types.BotCommand(command="start", description="Начать общение"),
    types.BotCommand(command="clear", description="Очистить историю диалога (короткую память)"),
    types.BotCommand(command="forget_me", description="Удалить все данные (и короткую, и долгосрочную память)")
]
```

**Что сделать:** Добавить запись для `/search`:
```python
commands = [
    types.BotCommand(command="start",     description="Начать общение"),
    types.BotCommand(command="search",    description="Поиск в интернете: /search <запрос>"),
    types.BotCommand(command="clear",     description="Очистить историю диалога"),
    types.BotCommand(command="forget_me", description="Удалить все мои данные"),
]
```

**Ожидаемый результат:** Команда `/search` видна в меню бота в Telegram.

---

### P1-6. Кэшировать содержимое `soul.md`

**Файл:** `bot.py`, функция `chat_handler`  
**Проблема:** Файл `soul.md` (системный промпт) открывается и читается заново при каждом входящем сообщении.

**Что сделать:**

1. Объявить константу уровня модуля. Разместить после объявления `dp = Dispatcher()`:
```python
# Загружаем "душу" бота один раз при старте
_soul_path = os.path.join(os.path.dirname(__file__), "soul.md")
try:
    with open(_soul_path, "r", encoding="utf-8") as _f:
        SOUL_CONTENT: str = _f.read()
except FileNotFoundError:
    SOUL_CONTENT = "Ты — умный ИИ-помощник с абсолютной долгосрочной памятью."
    logger.warning("soul.md не найден, используется промпт по умолчанию.")
```

2. Удалить блок чтения `soul.md` из `chat_handler` (~10 строк) и заменить на прямое использование константы `SOUL_CONTENT`.

**Ожидаемый результат:** `soul.md` читается ровно один раз при импорте модуля.

---

## Порядок выполнения

Рекомендуемая последовательность:

| # | Задача | Сложность | Риск регрессии |
|---|--------|-----------|----------------|
| 1 | P0-1 — `import time` | Тривиально | Нет |
| 2 | P0-2 — `sys.exit` при отсутствии токена | Тривиально | Нет |
| 3 | P1-5 — добавить `/search` в меню | Тривиально | Нет |
| 4 | P1-6 — кэшировать `soul.md` | Минимальная | Нет |
| 5 | P1-3 — переместить импорты | Минимальная | Низкий |
| 6 | P1-2 — callback на `create_task` | Небольшая | Нет |
| 7 | P1-4 — функция `make_chunks` | Средняя | Средний — нужно проверить все три места использования |
| 8 | P1-1 — глобальная `aiohttp.ClientSession` | Средняя | Средний — нужно убедиться, что сессия закрывается корректно |

---

## Критерии приёмки

- [ ] Бот запускается без `import time` и падает с `NameError` — **устранено**
- [ ] При отсутствии `TELEGRAM_BOT_TOKEN` процесс завершается с кодом 1 и понятным сообщением
- [ ] Ни одного `aiohttp.ClientSession()` внутри функций-обработчиков
- [ ] Ошибки `asyncio.create_task(add_to_vector_db(...))` логируются с трейсбеком
- [ ] Все три блока чанкинга заменены вызовом `make_chunks()`
- [ ] `import pypdf`, `import docx`, `from db.vector_db import get_embedding` находятся в начале файла
- [ ] `/search` присутствует в меню команд Telegram
- [ ] `soul.md` не открывается при каждом сообщении
- [ ] Все существующие сценарии работают: текст, голос, файл (.txt/.pdf/.docx), ссылка, `/search`, `/clear`, `/forget_me`
