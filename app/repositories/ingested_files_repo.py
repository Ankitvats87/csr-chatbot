from dataclasses import dataclass
from typing import Dict, List, Optional

from app.db.sqlite_client import SQLiteClient


@dataclass
class IngestedFile:
    file_id: str
    name: str
    modified_time: str
    last_indexed: Optional[str]
    n_chunks: int
    status: str


class IngestedFilesRepo:
    def __init__(self, db: SQLiteClient):
        self.db = db

    def all_indexed(self) -> Dict[str, IngestedFile]:
        rows = self.db.fetchall(
            "SELECT file_id, name, modified_time, last_indexed, n_chunks, status FROM ingested_files"
        )
        return {
            r["file_id"]: IngestedFile(
                file_id=r["file_id"],
                name=r["name"],
                modified_time=r["modified_time"],
                last_indexed=r["last_indexed"],
                n_chunks=r["n_chunks"] or 0,
                status=r["status"] or "pending",
            )
            for r in rows
        }

    def get(self, file_id: str) -> Optional[IngestedFile]:
        r = self.db.fetchone(
            "SELECT file_id, name, modified_time, last_indexed, n_chunks, status "
            "FROM ingested_files WHERE file_id = ?",
            (file_id,),
        )
        if not r:
            return None
        return IngestedFile(
            file_id=r["file_id"],
            name=r["name"],
            modified_time=r["modified_time"],
            last_indexed=r["last_indexed"],
            n_chunks=r["n_chunks"] or 0,
            status=r["status"] or "pending",
        )

    def upsert_success(self, file_id: str, name: str, modified_time: str, n_chunks: int) -> None:
        self.db.execute(
            """
            INSERT INTO ingested_files (file_id, name, modified_time, last_indexed, n_chunks, status)
            VALUES (?, ?, ?, datetime('now'), ?, 'indexed')
            ON CONFLICT(file_id) DO UPDATE SET
                name = excluded.name,
                modified_time = excluded.modified_time,
                last_indexed = excluded.last_indexed,
                n_chunks = excluded.n_chunks,
                status = 'indexed'
            """,
            (file_id, name, modified_time, n_chunks),
        )

    def mark_failed(self, file_id: str, name: str, modified_time: str) -> None:
        self.db.execute(
            """
            INSERT INTO ingested_files (file_id, name, modified_time, status)
            VALUES (?, ?, ?, 'failed')
            ON CONFLICT(file_id) DO UPDATE SET status = 'failed'
            """,
            (file_id, name, modified_time),
        )

    def delete(self, file_id: str) -> None:
        self.db.execute("DELETE FROM ingested_files WHERE file_id = ?", (file_id,))
