import asyncio
import os




from db.sqlite_db import init_db, save_message, get_history, clear_user_history
from db.vector_db import init_vector_db, add_to_vector_db, search_vectors, delete_user_vectors, get_embedding
from services.openrouter_client import call_llm, call_llm_stream, REWRITE_MODEL, MAIN_MODEL
from services.rerank import local_rerank
from bot import rewrite_query

async def main():
    print("=== Инициализация ===")
    await init_db()
    await init_vector_db()
    
    user_id = 99999
    
    print("\n=== Сценарий 1: Очистка старых данных ===")
    await clear_user_history(user_id)
    await delete_user_vectors(user_id)
    print("Память очищена.")
    
    print("\n=== Сценарий 2: Сохранение сообщений в SQLite ===")
    await save_message(user_id, "user", "Привет, я люблю играть в шахматы!")
    await save_message(user_id, "assistant", "Привет! Здорово, шахматы — отличная игра.")
    history = await get_history(user_id, 5)
    print(f"История из SQLite: {len(history)} сообщений.")
    
    print("\n=== Сценарий 3: Запись в VectorDB ===")
    await add_to_vector_db(user_id, "Пользователь сказал: Привет, я люблю играть в шахматы!\nОтвет бота: Привет! Здорово...")
    print("Вектор сохранен.")
    
    print("\n=== Сценарий 4: Rewrite Query ===")
    history_for_rewrite = [{"role": "user", "content": "Привет, я люблю играть в шахматы!"}]
    search_q = await rewrite_query("А как в них играть?", history_for_rewrite)
    print(f"Переписанный запрос: '{search_q}'")
    
    print("\n=== Сценарий 5: Поиск векторов ===")
    q_vec = await get_embedding(search_q)
    candidates = await search_vectors(user_id, q_vec, top_k=5)
    print(f"Найдено кандидатов: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"  [{i}]: {c[:50]}...")
        
    print("\n=== Сценарий 6: Rerank ===")
    candidates.append("Какой-то случайный текст про футбол.")
    candidates.append("Пользователь любит шахматы.")
    reranked = await local_rerank(search_q, candidates, top_n=2)
    print("Результат реранкинга:")
    for r in reranked:
        print(f"  -> {r[:50]}...")
        
    print("\n=== Сценарий 7: LLM Stream ===")
    messages = [{"role": "user", "content": "Расскажи короткий факт про шахматы."}]
    print("Ответ (стриминг): ", end="", flush=True)
    async for chunk in call_llm_stream(messages, MAIN_MODEL):
        print(chunk, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
