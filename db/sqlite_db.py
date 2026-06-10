import aiosqlite
import os

DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "chat_history.db")

async def init_db():
    os.makedirs(DB_DIR, exist_ok=True)
    async with aiosqlite.connect(DB_PATH, timeout=10.0) as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()

async def save_message(user_id: int, role: str, content: str) -> int:
    async with aiosqlite.connect(DB_PATH, timeout=10.0) as conn:
        cursor = await conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content)
        )
        await conn.commit()
        return cursor.lastrowid

async def get_history(user_id: int, limit: int = 5) -> list:
    async with aiosqlite.connect(DB_PATH, timeout=10.0) as conn:
        async with conn.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, id FROM messages 
                WHERE user_id = ? 
                ORDER BY id DESC 
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (user_id, limit)
        ) as cursor:
            rows = await cursor.fetchall()
            return [{"role": r, "content": c} for r, c in rows]

async def clear_user_history(user_id: int):
    async with aiosqlite.connect(DB_PATH, timeout=10.0) as conn:
        await conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await conn.commit()
