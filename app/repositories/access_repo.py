from dataclasses import dataclass
from typing import List, Optional

from app.db.sqlite_client import SQLiteClient


@dataclass
class AccessEntry:
    chat_id: int
    username: Optional[str]
    granted_by: Optional[int]
    granted_at: str
    revoked_at: Optional[str]


@dataclass
class AccessRequest:
    chat_id: int
    username: Optional[str]
    first_seen: str
    last_seen: str
    decision: Optional[str]


class AccessRepo:
    def __init__(self, db: SQLiteClient):
        self.db = db

    # ---------- allowlist ----------
    def is_active(self, chat_id: int) -> bool:
        row = self.db.fetchone(
            "SELECT revoked_at FROM access_list WHERE chat_id = ?",
            (chat_id,),
        )
        if not row:
            return False
        return row["revoked_at"] is None

    def grant(self, chat_id: int, username: Optional[str], granted_by: int) -> None:
        clean_username = username.lstrip("@").lower() if username else None
        if not clean_username:
            row = self.db.fetchone("SELECT username FROM access_requests WHERE chat_id = ?", (chat_id,))
            if row and row["username"]:
                clean_username = row["username"]
        self.db.execute(
            """
            INSERT INTO access_list (chat_id, username, granted_by)
            VALUES (?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
                username = COALESCE(excluded.username, access_list.username),
                granted_by = excluded.granted_by,
                granted_at = datetime('now'),
                revoked_at = NULL
            """,
            (chat_id, clean_username, granted_by),
        )

    def revoke(self, chat_id: int) -> bool:
        cur = self.db.execute(
            "UPDATE access_list SET revoked_at = datetime('now') "
            "WHERE chat_id = ? AND revoked_at IS NULL",
            (chat_id,),
        )
        return cur.rowcount > 0

    def list_active(self) -> List[AccessEntry]:
        rows = self.db.fetchall(
            "SELECT chat_id, username, granted_by, granted_at, revoked_at "
            "FROM access_list WHERE revoked_at IS NULL ORDER BY granted_at DESC"
        )
        return [
            AccessEntry(
                chat_id=r["chat_id"],
                username=r["username"],
                granted_by=r["granted_by"],
                granted_at=r["granted_at"],
                revoked_at=r["revoked_at"],
            )
            for r in rows
        ]

    def resolve_username(self, username: str) -> Optional[int]:
        """Find a chat_id by @username (matches allowlist and pending requests)."""
        u = username.lstrip("@").lower()
        row = self.db.fetchone(
            "SELECT chat_id FROM access_list WHERE LOWER(username) = ? "
            "UNION SELECT chat_id FROM access_requests WHERE LOWER(username) = ? LIMIT 1",
            (u, u),
        )
        return row["chat_id"] if row else None

    # ---------- access requests ----------
    def record_request(self, chat_id: int, username: Optional[str]) -> bool:
        """Returns True if this is a new request (first time we've seen them)."""
        clean_username = username.lstrip("@").lower() if username else None
        existing = self.db.fetchone(
            "SELECT chat_id FROM access_requests WHERE chat_id = ?", (chat_id,)
        )
        if existing:
            self.db.execute(
                "UPDATE access_requests SET last_seen = datetime('now'), "
                "username = COALESCE(?, username) WHERE chat_id = ?",
                (clean_username, chat_id),
            )
            return False
        self.db.execute(
            "INSERT INTO access_requests (chat_id, username) VALUES (?, ?)",
            (chat_id, clean_username),
        )
        return True

    def set_request_decision(self, chat_id: int, decision: str) -> None:
        self.db.execute(
            "UPDATE access_requests SET decision = ?, last_seen = datetime('now') "
            "WHERE chat_id = ?",
            (decision, chat_id),
        )

    def pending_requests(self) -> List[AccessRequest]:
        rows = self.db.fetchall(
            "SELECT chat_id, username, first_seen, last_seen, decision "
            "FROM access_requests WHERE decision IS NULL OR decision = 'pending' "
            "ORDER BY first_seen DESC"
        )
        return [
            AccessRequest(
                chat_id=r["chat_id"],
                username=r["username"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
                decision=r["decision"],
            )
            for r in rows
        ]
