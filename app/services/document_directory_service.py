"""Builds a structured directory of every ingested document so the LLM
can answer "list all meetings" style questions without depending on
similarity search.

Each indexed file is parsed for:
    - ordinal meeting number   (20th, 21st, …, 30th)
    - agenda date              (when the agenda was issued)
    - meeting date             (when the meeting was held)
    - document type            (Agenda, Minutes, BOD, Resolution, etc.)

The result is cached in-process and refreshed every N seconds (cheap —
just a SELECT + regex pass over ~20 filenames).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.db.sqlite_client import SQLiteClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Matches "20th", "21st", "22nd", "23rd", "26th" etc.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)

# Matches dates like 04.03.2025, 04-03-2025, 04/03/2025, 04.03.25
_DATE_RE = re.compile(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b")


@dataclass
class DirEntry:
    document_name: str
    n_chunks: int
    meeting_number: Optional[int]
    agenda_date: Optional[str]
    meeting_date: Optional[str]
    doc_type: str  # 'Agenda', 'Minutes', 'BOD', 'Resolution', 'Memorandum', 'Other'
    file_id: Optional[str] = None


def _ordinal(n: int) -> str:
    """13 -> '13th', 21 -> '21st', 23 -> '23rd'."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


class DocumentDirectoryService:
    REFRESH_SECONDS = 60

    def __init__(self, db: SQLiteClient):
        self.db = db
        self._cache: List[DirEntry] = []
        self._cached_at: float = 0.0

    def all(self) -> List[DirEntry]:
        if time.time() - self._cached_at > self.REFRESH_SECONDS:
            self._refresh()
        return self._cache

    def _refresh(self) -> None:
        rows = self.db.fetchall(
            "SELECT file_id, name, n_chunks FROM ingested_files WHERE status = 'indexed' AND n_chunks > 0"
        )
        entries: List[DirEntry] = []
        for r in rows:
            name = r["name"]
            entries.append(
                DirEntry(
                    document_name=name,
                    n_chunks=int(r["n_chunks"] or 0),
                    meeting_number=self._parse_ordinal(name),
                    agenda_date=self._parse_agenda_date(name),
                    meeting_date=self._parse_meeting_date(name),
                    doc_type=self._parse_doc_type(name),
                    file_id=r["file_id"],
                )
            )
        # Sort: numbered meetings ascending by ordinal, then non-numbered alphabetically.
        entries.sort(
            key=lambda e: (
                0 if e.meeting_number is not None else 1,
                e.meeting_number or 0,
                e.document_name,
            )
        )
        self._cache = entries
        self._cached_at = time.time()
        logger.info("document directory refreshed", extra={"n_files": len(entries)})

    # ---------- parsing helpers ----------
    @staticmethod
    def _parse_ordinal(name: str) -> Optional[int]:
        # We only care about the FIRST ordinal that looks like a meeting number
        # ("BOD minutes of 29 CSR Committee" → 29 from the raw "29 CSR").
        # Try the suffix form first ("26th", "20th").
        m = _ORDINAL_RE.search(name)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 999:
                    return n
            except ValueError:
                pass
        # Then plain "of N CSR" / "no. N" patterns.
        m2 = re.search(r"\b(?:of|no\.|number)\s+(\d{1,3})\b(?=\s+CSR)", name, re.IGNORECASE)
        if m2:
            try:
                return int(m2.group(1))
            except ValueError:
                pass
        return None

    @staticmethod
    def _parse_agenda_date(name: str) -> Optional[str]:
        # First date in the filename after "Agenda dt" / "Agenda dated".
        # Allow no-space variants like "Agenda dated18.08.2023".
        m = re.search(r"Agenda\s+(?:dt|dated)\s*[._\-]*\s*([0-9]{1,2}[.\-/][0-9]{1,2}[.\-/][0-9]{2,4})", name, re.IGNORECASE)
        if m:
            d = _DATE_RE.search(m.group(1))
            if d:
                return DocumentDirectoryService._normalise_date(d.group(0))
        return None

    @staticmethod
    def _parse_meeting_date(name: str) -> Optional[str]:
        m = re.search(r"Meeting\s+dated?[_\s]*([0-9.\-/]+)", name, re.IGNORECASE)
        if m:
            d = _DATE_RE.search(m.group(0))
            if d:
                return DocumentDirectoryService._normalise_date(d.group(0))
        # "BOD minutes of 29 CSR Committee dated 04.12.2025" pattern.
        m2 = re.search(r"dated?\s+([0-9.\-/]+)", name, re.IGNORECASE)
        if m2:
            d = _DATE_RE.search(m2.group(0))
            if d:
                return DocumentDirectoryService._normalise_date(d.group(0))
        return None

    @staticmethod
    def _normalise_date(s: str) -> str:
        m = _DATE_RE.search(s)
        if not m:
            return s
        dd, mm, yy = m.group(1), m.group(2), m.group(3)
        if len(yy) == 2:
            yy = "20" + yy
        return f"{int(dd):02d}.{int(mm):02d}.{yy}"

    @staticmethod
    def _parse_doc_type(name: str) -> str:
        n = name.lower()
        if "bod" in n or "board" in n or "memorandum" in n:
            return "BOD/Memorandum"
        if "resolution" in n or "circulation" in n:
            return "Resolution"
        if "minutes" in n:
            return "Minutes"
        if "agenda" in n:
            return "Agenda"
        return "Other"

    # ---------- formatting for the prompt ----------
    def format_for_prompt(self) -> str:
        entries = self.all()
        if not entries:
            return "(no documents indexed yet)"
        lines: List[str] = []
        for e in entries:
            parts: List[str] = []
            if e.meeting_number is not None:
                parts.append(f"Meeting #{e.meeting_number}")
            parts.append(f"[{e.doc_type}]")
            if e.agenda_date:
                parts.append(f"agenda {e.agenda_date}")
            if e.meeting_date and e.meeting_date != e.agenda_date:
                parts.append(f"held {e.meeting_date}")
            parts.append(f"chunks={e.n_chunks}")
            parts.append(f'"{e.document_name}"')
            lines.append("  - " + ", ".join(parts))
        return "\n".join(lines)

    def document_names_for_meeting(self, ordinal: int) -> List[str]:
        return [e.document_name for e in self.all() if e.meeting_number == ordinal]

    # ---------- human-readable source labels ----------
    @staticmethod
    def _friendly_label(e: DirEntry) -> str:
        """Turn a directory entry into a citation label a Telegram user understands,
        e.g. '23rd CSR Committee Minutes (held 22.07.2024)'. Falls back to the raw
        readable filename when no meeting ordinal could be parsed."""
        if e.meeting_number is not None:
            label = f"{_ordinal(e.meeting_number)} CSR Committee {e.doc_type}"
            date = e.meeting_date or e.agenda_date
            if date:
                label += f" (held {date})" if e.meeting_date else f" (agenda {date})"
            return label
        return e.document_name

    def humanize_source(self, raw: str) -> str:
        """Map a raw Pinecone source label (a Drive file-id like
        '1Nqw9r7Ozt_8Dy3MBiL4xnc2slbRwcKTF.pdf') to a readable citation.
        Leaves already-readable labels (e.g. project-master names) untouched."""
        if not raw:
            return raw
        s = raw.strip()
        # Drop any extension so '<file_id>.pdf'/'.json' resolves to the file_id.
        base = re.sub(r"\.(pdf|json|docx?|txt)$", "", s, flags=re.IGNORECASE)
        by_id = {e.file_id: e for e in self.all() if e.file_id}
        entry = by_id.get(base) or by_id.get(s)
        if entry:
            return self._friendly_label(entry)
        # Maybe the label already equals a readable document_name.
        for e in self.all():
            if e.document_name == s:
                return self._friendly_label(e)
        return s

    @staticmethod
    def clean_page(page) -> Optional[str]:
        """Normalize a page value: '3.0' -> '3', None -> None."""
        if page is None or page == "":
            return None
        s = str(page).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s or None
