import os
import sqlite3
import threading
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER NOT NULL,
    role        TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT    NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_turns_chat_created ON conversation_turns(chat_id, created_at);

CREATE TABLE IF NOT EXISTS request_logs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    question     TEXT,
    answer       TEXT,
    n_retrieved  INTEGER,
    latency_ms   INTEGER,
    model        TEXT,
    provider     TEXT,
    status       TEXT,
    error        TEXT,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_logs_chat_created ON request_logs(chat_id, created_at);

-- Reserved for Phase 2 (ingestion + access control). Created early so the
-- swap to Postgres later requires no schema-migration code change.
CREATE TABLE IF NOT EXISTS ingested_files (
    file_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    modified_time  TEXT NOT NULL,
    last_indexed   TEXT,
    n_chunks       INTEGER DEFAULT 0,
    status         TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS access_list (
    chat_id     INTEGER PRIMARY KEY,
    username    TEXT,
    granted_by  INTEGER,
    granted_at  TEXT NOT NULL DEFAULT (datetime('now')),
    revoked_at  TEXT
);

CREATE TABLE IF NOT EXISTS access_requests (
    chat_id     INTEGER PRIMARY KEY,
    username    TEXT,
    first_seen  TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen   TEXT NOT NULL DEFAULT (datetime('now')),
    decision    TEXT
);

CREATE TABLE IF NOT EXISTS web_admin_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteClient:
    """Thin wrapper providing a thread-safe connection.
    Repositories take this client (DI) so swapping to asyncpg later only
    requires changing this file + the repo implementations.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._conn = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; explicit txns where needed
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(_SCHEMA)
        logger.info("sqlite connected", extra={"path": self.path})

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteClient not connected. Call connect() first.")
        return self._conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, params)

    def fetchall(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchall()

    def fetchone(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            return cur.fetchone()

    def health_ok(self) -> bool:
        try:
            self.fetchone("SELECT 1")
            return True
        except Exception:
            return False
