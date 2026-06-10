from sentence_transformers import CrossEncoder
import asyncio
import os
import logging

logger = logging.getLogger(__name__)

# Путь для кэширования моделей внутри Docker-контейнера
MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

logger.info(f"Загрузка локального реранкера {MODEL_NAME}...")
try:
    if os.path.exists("local_model"):
        logger.info("Используется закэшированная модель из ./local_model")
        reranker_model = CrossEncoder("local_model")
    else:
        reranker_model = CrossEncoder(MODEL_NAME)
    logger.info("Реранкер успешно загружен.")
except Exception as e:
    logger.error(f"Ошибка загрузки реранкера: {e}")
    reranker_model = None

async def local_rerank(query: str, documents: list[str], top_n: int = 4) -> list[str]:
    """Локальный реранкинг кандидатов с помощью Cross-Encoder модели"""
    if not documents or not reranker_model:
        return documents[:top_n]
    
    try:
        loop = asyncio.get_event_loop()
        pairs = [[query, doc] for doc in documents]
        
        # sentence-transformers.predict блокирует поток, запускаем в thread pool
        scores = await loop.run_in_executor(None, reranker_model.predict, pairs)
        
        # Сортируем документы по убыванию скора
        scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        
        logger.info(f"Реранкер отфильтровал {len(documents)} кандидатов до {top_n}")
        return [doc for doc, score in scored_docs[:top_n]]
    except Exception as e:
        logger.error(f"Ошибка во время реранкинга: {e}")
        return documents[:top_n]
