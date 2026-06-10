from typing import Optional

from app.db.sqlite_client import SQLiteClient


class RequestLogRepo:
    def __init__(self, db: SQLiteClient):
        self.db = db

    def log(
        self,
        chat_id: int,
        question: str,
        answer: Optional[str],
        n_retrieved: int,
        latency_ms: int,
        model: str,
        provider: str,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO request_logs
                (chat_id, question, answer, n_retrieved, latency_ms, model, provider, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chat_id, question, answer, n_retrieved, latency_ms, model, provider, status, error),
        )
