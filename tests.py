import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db.vector_db import add_chunks_to_vector_db, search_vectors, init_vector_db
from bot import make_chunks, rewrite_query
from services.rerank import local_rerank
from services.openrouter_client import call_llm, describe_image

async def test_text_splitting():
    print("\n--- ТЕСТ 1: Разбиение текста на чанки (make_chunks) ---")
    long_text = "А" * 2000
    chunks = make_chunks(long_text, "Тестовый источник")
    print(f"Размер исходного текста: {len(long_text)} символов.")
    print(f"Количество чанков: {len(chunks)}")
    assert len(chunks) > 1, "Текст должен разбиваться на несколько чанков"
    assert chunks[0].startswith("Тестовый источник:"), "Чанк должен начинаться с названия источника"
    print("✅ ТЕСТ 1 ПРОЙДЕН")

async def test_query_rewriting():
    print("\n--- ТЕСТ 2: Переписывание запроса LLM (rewrite_query) ---")
    # Эмулируем историю, где пользователь говорит "он", имея в виду Пушкина
    chat_history = [
        {"role": "user", "content": "Кто написал Евгения Онегина?"},
        {"role": "assistant", "content": "Александр Сергеевич Пушкин."}
    ]
    user_message = "А в каком году он родился?"
    rewritten = await rewrite_query(user_message, chat_history)
    print(f"Оригинальный запрос: '{user_message}'")
    print(f"Переписанный запрос: '{rewritten}'")
    assert "пушкин" in rewritten.lower() or "онегин" in rewritten.lower(), "LLM не подставила контекст из истории"
    print("✅ ТЕСТ 2 ПРОЙДЕН")

async def test_rag_pipeline():
    print("\n--- ТЕСТ 3: Гибридный поиск и Реранкер (RAG Pipeline) ---")
    await init_vector_db()
    test_user_id = 888888888
    
    chunks = [
        "Документ 1: В 2024 году выйдет новая модель iPhone 16.",
        "Документ 2: Автомобили Tesla используют камеры для автопилота.",
        "Документ 3: Компания Apple разрабатывает собственные процессоры M3."
    ]
    
    print("Индексация чанков...")
    await add_chunks_to_vector_db(test_user_id, chunks)
    await asyncio.sleep(1) # Ждем синхронизации
    
    query = "Какие процессоры делает Эпл?"
    print(f"Запрос: '{query}'")
    candidates = await search_vectors(test_user_id, query, top_k=5)
    print(f"Кандидатов от гибридного поиска: {len(candidates)}")
    
    relevant = await local_rerank(query, candidates, top_n=1)
    print(f"Победитель после реранкера: {relevant[0]}")
    assert "M3" in relevant[0], "Реранкер выбрал не тот документ"
    print("✅ ТЕСТ 3 ПРОЙДЕН")

async def test_vision():
    print("\n--- ТЕСТ 4: Генерация описания картинки (Vision LLM) ---")
    # Прозрачный 1x1 pixel PNG в base64
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    print("Отправка тестовой картинки в OpenRouter Vision...")
    description = await describe_image(tiny_png_b64)
    print(f"Ответ Vision-модели (первые 100 символов): {description[:100]}...")
    assert len(description) > 5, "Vision-модель вернула слишком короткое или пустое описание"
    print("✅ ТЕСТ 4 ПРОЙДЕН")

async def run_all_tests():
    print("🚀 ЗАПУСК ГРУППЫ БАЗОВЫХ ТЕСТОВ СЦЕНАРИЕВ\n")
    try:
        await test_text_splitting()
        await test_query_rewriting()
        await test_rag_pipeline()
        await test_vision()
        
        print("\n🎉 ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
        print("\nПРИМЕЧАНИЕ ПО ВОЗМОЖНОСТЯМ:")
        print("- Чтение файлов: Поддерживается (.pdf, .docx, .txt), код в bot.py использует pypdf/python-docx/Jina Reader.")
        print("- Распознавание аудио: Поддерживается (Groq Whisper API).")
        print("- Чтение URL: Поддерживается (Jina Reader).")
        print("- Распознавание картинок: Поддерживается (OpenRouter Vision LLM).")
    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
