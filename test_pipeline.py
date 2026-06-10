import asyncio
import os
import sys

# Добавляем корневую папку в sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.vector_db import add_chunks_to_vector_db, search_vectors, init_vector_db
from bot import local_rerank

async def run_tests():
    print("🚀 Запуск интеграционного тестирования RAG пайплайна...")
    
    # 1. Инициализация БД
    print("\n1. Инициализация Vector DB...")
    await init_vector_db()
    
    # Фиктивный user_id для тестов
    test_user_id = 999999999
    
    # 2. Подготовка сложных тестовых данных
    # Специально делаем так, чтобы векторный поиск давал один результат,
    # а BM25 (по ключам) - другой, чтобы проверить гибридность и реранкер.
    chunks = [
        "Фрагмент 1: Илон Маск купил компанию Twitter за 44 миллиарда долларов в 2022 году.",
        "Фрагмент 2: В 2022 году была выпущена новая модель автомобиля Tesla Model X.",
        "Фрагмент 3: Социальная сеть Twitter сменила логотип на букву X.",
        "Фрагмент 4: Марк Цукерберг основал Meta и владеет Instagram.",
        "Фрагмент 5: Космическая компания SpaceX успешно запустила ракету Falcon Heavy 44 раза."
    ]
    
    print(f"\n2. Индексирование {len(chunks)} тестовых чанков...")
    await add_chunks_to_vector_db(test_user_id, chunks)
    print("✅ Чанки проиндексированы (Dense + Sparse/BM25).")
    
    # Даем Qdrant секунду на обновление индексов
    await asyncio.sleep(1)
    
    # 3. Тест 1: Точный поиск (Число и имя) - BM25 должен помочь
    query = "За сколько Илон Маск купил Twitter?"
    print(f"\n3. Тестирование гибридного поиска. Запрос: '{query}'")
    
    candidates = await search_vectors(test_user_id, query, top_k=5)
    print(f"🔍 Найдено кандидатов (Hybrid - RRF): {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"  - [{i+1}] {c}")
        
    print("\n4. Тестирование локального реранкера (DiTy)...")
    relevant = await local_rerank(query, candidates, top_n=2)
    print(f"🎯 Топ-2 после реранкера:")
    for i, r in enumerate(relevant):
        print(f"  - [{i+1}] {r}")
        
    assert len(relevant) > 0 and "44 миллиарда" in relevant[0], "Реранкер не вывел правильный ответ на первое место!"
    print("\n✅ Тест гибридного RAG пайплайна (Dense + Sparse + Reranker) пройден успешно!")

if __name__ == "__main__":
    asyncio.run(run_tests())
