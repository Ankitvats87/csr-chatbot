from typing import List

from app.db.sqlite_client import SQLiteClient
from app.models.message_model import Turn


class MemoryRepo:
    def __init__(self, db: SQLiteClient):
        self.db = db

    def append(self, chat_id: int, role: str, content: str) -> None:
        self.db.execute(
            "INSERT INTO conversation_turns (chat_id, role, content) VALUES (?, ?, ?)",
            (chat_id, role, content),
        )

    def recent(self, chat_id: int, window: int) -> List[Turn]:
        rows = self.db.fetchall(
            """
            SELECT role, content FROM (
                SELECT role, content, id FROM conversation_turns
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (chat_id, window * 2),  # 10 turns = up to 20 messages (user+assistant pairs)
        )
        return [Turn(role=r["role"], content=r["content"]) for r in rows]

    def clear(self, chat_id: int) -> None:
        self.db.execute("DELETE FROM conversation_turns WHERE chat_id = ?", (chat_id,))
