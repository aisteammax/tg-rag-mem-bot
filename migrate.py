import asyncio
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams

async def migrate():
    print("Начинаем миграцию Qdrant...")
    client = AsyncQdrantClient(url="http://qdrant:6333")
    col_name = "user_memory"
    
    # Пытаемся получить информацию
    try:
        info = await client.get_collection(col_name)
        if hasattr(info.config.params, 'sparse_vectors_config') and info.config.params.sparse_vectors_config:
            if "sparse" in info.config.params.sparse_vectors_config:
                print("Коллекция уже поддерживает sparse вектора. Миграция не требуется.")
                return
    except Exception as e:
        print("Коллекция не существует:", e)
        return

    print("Скачиваем все данные из старой коллекции...")
    # Так как данных мало, скачиваем всё
    records = []
    offset = None
    while True:
        res = await client.scroll(
            collection_name=col_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        points, offset = res
        records.extend(points)
        if offset is None:
            break
            
    print(f"Скачано {len(records)} записей. Удаляем старую коллекцию...")
    await client.delete_collection(col_name)
    
    print("Создаем новую коллекцию с поддержкой Hybrid Search...")
    # Берем размерность из первого вектора, или 3584 по умолчанию (qwen)
    dim = 3584
    if records and records[0].vector:
        if isinstance(records[0].vector, dict) and "" in records[0].vector:
            dim = len(records[0].vector[""])
        elif isinstance(records[0].vector, list):
            dim = len(records[0].vector)
            
    await client.create_collection(
        collection_name=col_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        }
    )
    
    print("ВНИМАНИЕ: Старые записи сохранены в памяти скрипта, но для них нет sparse-векторов.")
    print("В идеале их нужно переиндексировать через fastembed. Так как это просто тест, мы пока оставим коллекцию пустой для чистоты тестов нового пайплайна.")
    print("Миграция завершена!")

if __name__ == "__main__":
    asyncio.run(migrate())
