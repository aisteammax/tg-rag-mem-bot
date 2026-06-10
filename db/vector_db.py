from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import AsyncOpenAI
import os
import logging
import uuid
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# OpenRouter Client for embeddings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "qwen/qwen3-embedding-8b")
embed_client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)

# Async Qdrant Client
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_client = AsyncQdrantClient(url=QDRANT_URL)

COLLECTION_NAME = "user_memory"

try:
    from fastembed import SparseTextEmbedding
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
except ImportError:
    logger.warning("fastembed не установлен. Гибридный поиск будет работать только в режиме плотных векторов.")
    sparse_model = None

def get_sparse_embedding(text: str):
    if not sparse_model:
        return None
    res = list(sparse_model.embed([text]))[0]
    return {"indices": res.indices.tolist(), "values": res.values.tolist()}

def get_sparse_embeddings_batch(texts: list[str]):
    if not sparse_model:
        return [None] * len(texts)
    res_list = list(sparse_model.embed(texts))
    return [{"indices": r.indices.tolist(), "values": r.values.tolist()} for r in res_list]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_embedding(text: str) -> list[float]:
    """Генерация эмбеддинга через OpenRouter API"""
    try:
        response = await embed_client.embeddings.create(
            input=[text],
            model=EMBED_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Ошибка получения эмбеддинга от OpenRouter ({EMBED_MODEL}): {e}")
        raise e

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Пакетная генерация эмбеддингов"""
    try:
        response = await embed_client.embeddings.create(
            input=texts,
            model=EMBED_MODEL
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]
    except Exception as e:
        logger.error(f"Ошибка пакетного получения эмбеддингов ({EMBED_MODEL}): {e}")
        raise e

async def init_vector_db():
    """Инициализация Qdrant коллекции с динамическим определением размерности эмбеддинга"""
    try:
        from qdrant_client import models
        collections_res = await qdrant_client.get_collections()
        collections = collections_res.collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            logger.info("Коллекция не найдена. Получаем тестовый эмбеддинг для определения размерности...")
            test_vector = await get_embedding("тест")
            dimension = len(test_vector)
            logger.info(f"Определена размерность вектора: {dimension}. Создаем коллекцию '{COLLECTION_NAME}'...")
            
            await qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(on_disk=False)
                    )
                }
            )
            logger.info(f"Коллекция '{COLLECTION_NAME}' успешно создана.")
        else:
            # Пытаемся обновить коллекцию, добавив поддержку sparse векторов (если её нет)
            try:
                await qdrant_client.update_collection(
                    collection_name=COLLECTION_NAME,
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(
                            index=models.SparseIndexParams(on_disk=False)
                        )
                    }
                )
            except Exception as e:
                pass # Уже существует или не поддерживается
    except Exception as e:
        logger.error(f"Не удалось инициализировать векторную базу данных: {e}")

async def add_to_vector_db(user_id: int, text: str):
    """Добавление текста в векторное хранилище с фильтрацией по user_id"""
    try:
        from qdrant_client import models
        dense_vector = await get_embedding(text)
        sparse_vector_data = get_sparse_embedding(text)
        vector_id = str(uuid.uuid4())
        
        vector_payload = {"": dense_vector}
        if sparse_vector_data:
            vector_payload["sparse"] = models.SparseVector(
                indices=sparse_vector_data["indices"],
                values=sparse_vector_data["values"]
            )
            
        await qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=vector_id,
                    vector=vector_payload,
                    payload={"user_id": user_id, "text": text}
                )
            ]
        )
        logger.info(f"Запись {vector_id} успешно проиндексирована в Vector DB для пользователя {user_id}")
    except Exception as e:
        logger.error(f"Ошибка записи в Vector DB: {e}")

async def add_chunks_to_vector_db(user_id: int, chunks: list[str], batch_size: int = 20):
    """Пакетное добавление чанков в векторное хранилище"""
    from qdrant_client import models
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            dense_vectors = await get_embeddings_batch(batch)
            sparse_vectors_data = get_sparse_embeddings_batch(batch)
            points = []
            for j, text in enumerate(batch):
                vector_id = str(uuid.uuid4())
                vector_payload = {"": dense_vectors[j]}
                
                if sparse_vectors_data[j]:
                    vector_payload["sparse"] = models.SparseVector(
                        indices=sparse_vectors_data[j]["indices"],
                        values=sparse_vectors_data[j]["values"]
                    )
                    
                points.append(
                    models.PointStruct(
                        id=vector_id,
                        vector=vector_payload,
                        payload={"user_id": user_id, "text": text}
                    )
                )
            await qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            logger.info(f"Успешно проиндексирован батч из {len(batch)} чанков для пользователя {user_id}")
        except Exception as e:
            logger.error(f"Ошибка батчевой записи в Vector DB: {e}")

async def search_vectors(user_id: int, query_text: str, top_k: int = 20) -> list[str]:
    """Гибридный поиск схожих текстов в Qdrant с фильтром по user_id"""
    try:
        from qdrant_client import models
        dense_vector = await get_embedding(query_text)
        sparse_vector_data = get_sparse_embedding(query_text)
        
        prefetch = []
        # Плотный векторный поиск (семантика)
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using="",
                limit=top_k,
            )
        )
        
        # Разреженный векторный поиск (BM25 - ключевые слова)
        if sparse_vector_data:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector_data["indices"],
                        values=sparse_vector_data["values"]
                    ),
                    using="sparse",
                    limit=top_k,
                )
            )
        
        search_result = await qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id)
                    )
                ]
            ),
            limit=top_k
        )
        return [hit.payload["text"] for hit in search_result.points]
    except Exception as e:
        logger.error(f"Ошибка поиска в Vector DB: {e}")
        return []

async def delete_user_vectors(user_id: int):
    """Удаление всех векторов пользователя"""
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        await qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            )
        )
        logger.info(f"Все векторы для пользователя {user_id} удалены.")
    except Exception as e:
        logger.error(f"Ошибка удаления векторов в Vector DB: {e}")
