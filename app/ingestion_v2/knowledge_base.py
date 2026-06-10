"""CSR Knowledge Base — the canonical relational layer.

The corpus is fundamentally relational: ~11 meetings, ~35 recurring projects,
a stable attendee group, budgets per FY. This module turns per-document
extraction JSONs into clean SQLite tables so factual/aggregate questions are
answered EXACTLY (SQL), with vector search reserved for narrative questions.

Tables (created in the existing app.db, all prefixed kb_):

  kb_meetings        one row per CSR meeting        (no, dates, FY, source doc)
  kb_attendance      person × meeting               (canonical person names)
  kb_projects        one row per canonical project  (~35, letter labels excluded)
  kb_project_events  project × meeting lifecycle    (stage, amounts, source)
  kb_budget_lines    every monetary figure          (FY, line item, amount, kind)
  kb_decisions       agenda item decisions          (item no e.g. '26.3', text)

The whole KB is REBUILT from data/processed_v2/*.json on every ingestion run
(idempotent, <1s for this corpus size) — no incremental-update bugs possible.
Every row carries its source document for citations.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

from app.db.sqlite_client import SQLiteClient
from app.ingestion_v2.project_master_builder import _canon_key, is_generic_label
from app.ingestion_v2.schemas import ExtractedDocument
from app.utils.logger import get_logger

logger = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS kb_meetings (
    meeting_no   INTEGER PRIMARY KEY,
    meeting_date TEXT,
    agenda_date  TEXT,
    financial_year TEXT,
    committee    TEXT,
    source_docs  TEXT
);
CREATE TABLE IF NOT EXISTS kb_attendance (
    meeting_no   INTEGER,
    person       TEXT,
    designation  TEXT,
    role         TEXT,
    source_doc   TEXT,
    UNIQUE(meeting_no, person, role)
);
CREATE TABLE IF NOT EXISTS kb_projects (
    project_key  TEXT PRIMARY KEY,
    project_name TEXT,
    ngo_name     TEXT,
    sector       TEXT,
    schedule_vii TEXT,
    state        TEXT,
    district     TEXT,
    status       TEXT,
    approved_cost REAL,
    disbursed_amount REAL,
    balance_amount REAL,
    beneficiary_type TEXT,
    source_docs  TEXT
);
CREATE TABLE IF NOT EXISTS kb_project_events (
    project_key  TEXT,
    meeting_no   INTEGER,
    stage        TEXT,
    status       TEXT,
    approved_cost REAL,
    disbursed_amount REAL,
    notes        TEXT,
    source_doc   TEXT
);
CREATE TABLE IF NOT EXISTS kb_budget_lines (
    financial_year TEXT,
    meeting_no   INTEGER,
    line_item    TEXT,
    amount       REAL,
    kind         TEXT,
    source_doc   TEXT
);
CREATE TABLE IF NOT EXISTS kb_decisions (
    meeting_no   INTEGER,
    agenda_item  TEXT,
    title        TEXT,
    decision_text TEXT,
    project_name TEXT,
    amount       REAL,
    source_doc   TEXT
);
"""

_TABLES = [
    "kb_meetings", "kb_attendance", "kb_projects",
    "kb_project_events", "kb_budget_lines", "kb_decisions",
]

_SALUTATION_RE = re.compile(
    r"^\s*(shri|smt\.?|dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|sh\.?)\s+", re.IGNORECASE
)


def canonical_person(name: str) -> str:
    """Normalize a person name for cross-meeting identity: strip salutations,
    collapse whitespace, title-case."""
    s = _SALUTATION_RE.sub("", (name or "").strip())
    s = re.sub(r"\s+", " ", s)
    return s.title()


class KnowledgeBase:
    def __init__(self, db: SQLiteClient):
        self.db = db

    # ── build ──────────────────────────────────────────────────────────
    def rebuild(self, processed_dir: Path, doc_name_by_file_id: Dict[str, str]) -> dict:
        """Full rebuild from extraction JSONs. Returns row counts per table."""
        con = self.db.conn
        con.executescript(_DDL)
        for t in _TABLES:
            con.execute(f"DELETE FROM {t}")

        docs: List[tuple[str, ExtractedDocument]] = []
        for f in sorted(processed_dir.glob("*.json")):
            try:
                extracted = ExtractedDocument.model_validate_json(
                    f.read_text(encoding="utf-8")
                )
                doc_name = doc_name_by_file_id.get(f.stem, f.stem)
                docs.append((doc_name, extracted))
            except Exception as e:
                logger.warning("kb: failed to load extraction json", extra={"file": f.name, "err": str(e)})

        # Pass 1 — meetings (merge agenda + minutes rows for the same ordinal).
        meetings: Dict[int, dict] = {}
        for doc_name, ex in docs:
            mnum = ex.meeting.meeting_number
            if mnum is None:
                continue
            m = meetings.setdefault(mnum, {"meeting_date": None, "financial_year": None, "committee": None, "docs": []})
            m["docs"].append(doc_name)
            if ex.meeting.meeting_date and not m["meeting_date"]:
                m["meeting_date"] = ex.meeting.meeting_date
            if ex.meeting.financial_year and not m["financial_year"]:
                m["financial_year"] = ex.meeting.financial_year
            if ex.governance.committee_name and not m["committee"]:
                m["committee"] = ex.governance.committee_name
        for mnum, m in meetings.items():
            con.execute(
                "INSERT OR REPLACE INTO kb_meetings (meeting_no, meeting_date, agenda_date, financial_year, committee, source_docs) VALUES (?,?,?,?,?,?)",
                (mnum, m["meeting_date"], None, m["financial_year"], m["committee"], "; ".join(dict.fromkeys(m["docs"]))),
            )

        # Pass 2 — attendance / decisions / budget lines.
        for doc_name, ex in docs:
            mnum = ex.meeting.meeting_number
            for a in ex.attendees:
                person = canonical_person(a.name)
                if not person or _SALUTATION_RE.match(person):
                    continue
                con.execute(
                    "INSERT OR IGNORE INTO kb_attendance (meeting_no, person, designation, role, source_doc) VALUES (?,?,?,?,?)",
                    (mnum, person, a.designation, (a.role or "member").lower(), doc_name),
                )
            for d in ex.decisions:
                con.execute(
                    "INSERT INTO kb_decisions (meeting_no, agenda_item, title, decision_text, project_name, amount, source_doc) VALUES (?,?,?,?,?,?,?)",
                    (mnum, d.agenda_item, d.title, d.decision_text,
                     None if is_generic_label(d.project_name) else d.project_name,
                     d.amount, doc_name),
                )
            for b in ex.budget_lines:
                if is_generic_label(b.line_item):
                    continue
                con.execute(
                    "INSERT INTO kb_budget_lines (financial_year, meeting_no, line_item, amount, kind, source_doc) VALUES (?,?,?,?,?,?)",
                    (b.financial_year, mnum, b.line_item, b.amount, (b.kind or "").lower(), doc_name),
                )

        # Pass 3 — canonical projects + lifecycle events.
        projects: Dict[str, dict] = {}
        for doc_name, ex in docs:
            mnum = ex.meeting.meeting_number
            for p in ex.projects:
                if is_generic_label(p.project_name):
                    continue
                key = _canon_key(p.project_name)
                if not key:
                    continue
                pr = projects.setdefault(key, {
                    "name": p.project_name, "ngo": None, "sector": None, "schedule": None,
                    "state": None, "district": None, "status": "Unknown",
                    "approved": None, "disbursed": None, "balance": None,
                    "beneficiary_type": None, "docs": [], "max_meeting": -1,
                })
                pr["docs"].append(doc_name)
                # Longest name wins (most descriptive variant of the ~35 recurring projects).
                if len(p.project_name) > len(pr["name"]):
                    pr["name"] = p.project_name
                if p.ngo and p.ngo.ngo_name and not pr["ngo"]:
                    pr["ngo"] = p.ngo.ngo_name
                if p.classification:
                    pr["sector"] = pr["sector"] or p.classification.csr_sector
                    pr["schedule"] = pr["schedule"] or p.classification.csr_schedule
                if p.geography:
                    pr["state"] = pr["state"] or p.geography.state
                    pr["district"] = pr["district"] or p.geography.district
                if p.beneficiary and p.beneficiary.beneficiary_type:
                    pr["beneficiary_type"] = pr["beneficiary_type"] or p.beneficiary.beneficiary_type
                fin = p.financial
                if fin:
                    pr["approved"] = pr["approved"] or fin.approved_cost or fin.project_cost
                    if fin.disbursed_amount is not None:
                        pr["disbursed"] = fin.disbursed_amount
                    if fin.balance_amount is not None:
                        pr["balance"] = fin.balance_amount
                # Latest meeting's status wins (status auto-update on new ingestion).
                m_for_status = mnum if mnum is not None else -1
                if p.project_status.value != "Unknown" and m_for_status >= pr["max_meeting"]:
                    pr["status"] = p.project_status.value
                    pr["max_meeting"] = m_for_status

                con.execute(
                    "INSERT INTO kb_project_events (project_key, meeting_no, stage, status, approved_cost, disbursed_amount, notes, source_doc) VALUES (?,?,?,?,?,?,?,?)",
                    (key, mnum, p.lifecycle_stage.value, p.project_status.value,
                     (fin.approved_cost or fin.project_cost) if fin else None,
                     fin.disbursed_amount if fin else None,
                     p.notes, doc_name),
                )

        for key, pr in projects.items():
            con.execute(
                "INSERT OR REPLACE INTO kb_projects (project_key, project_name, ngo_name, sector, schedule_vii, state, district, status, approved_cost, disbursed_amount, balance_amount, beneficiary_type, source_docs) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, pr["name"], pr["ngo"], pr["sector"], pr["schedule"], pr["state"],
                 pr["district"], pr["status"], pr["approved"], pr["disbursed"], pr["balance"],
                 pr["beneficiary_type"], "; ".join(dict.fromkeys(pr["docs"]))),
            )

        con.commit()
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
        logger.info("knowledge base rebuilt", extra=counts)
        return counts
