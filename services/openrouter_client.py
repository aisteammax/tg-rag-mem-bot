from openai import AsyncOpenAI
import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# OpenRouter Client Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"

MAIN_MODEL = os.getenv("MAIN_MODEL", "deepseek/deepseek-chat")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", "deepseek/deepseek-chat")

client = None
if OPENROUTER_API_KEY:
    client = AsyncOpenAI(base_url=OPENROUTER_URL, api_key=OPENROUTER_API_KEY)
else:
    logger.warning("OPENROUTER_API_KEY не установлен! Пожалуйста, добавьте его в .env")

def get_client():
    global client
    if not client:
        key = os.getenv("OPENROUTER_API_KEY")
        if key:
            client = AsyncOpenAI(base_url=OPENROUTER_URL, api_key=key)
        else:
            raise ValueError("OPENROUTER_API_KEY не задан.")
    return client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
async def call_llm(messages: list[dict], model: str) -> str:
    """Запрос к LLM через OpenRouter с передачей списка сообщений (с retry)"""
    c = get_client()
    try:
        response = await c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            extra_headers={
                "HTTP-Referer": "https://github.com/google/antigravity",
                "X-Title": "TG RAG Bot",
            }
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка запроса к OpenRouter ({model}): {e}")
        raise e

async def call_llm_stream(messages: list[dict], model: str):
    """Стриминговый запрос к LLM через OpenRouter"""
    c = get_client()
    try:
        response = await c.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            stream=True,
            extra_headers={
                "HTTP-Referer": "https://github.com/google/antigravity",
                "X-Title": "TG RAG Bot",
            }
        )
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"Ошибка стримингового запроса к OpenRouter ({model}): {e}")
        raise e
