"""Workflow-specialist fetch layer over the CSR Knowledge Base.

For each query class the intelligence layer asks this service for the
relevant AUTHORITATIVE structured blocks (exact SQL results formatted as
markdown, with source documents). These blocks are injected at the TOP of
the generation context, above vector-retrieved chunks, so factual questions
are answered from exact data — the LLM only formats, it cannot hallucinate
a number that isn't in the block.

All methods fail soft: if the KB tables don't exist yet (first boot before
ingestion), they return "" and the system falls back to vector retrieval.
"""
from __future__ import annotations

import re
from typing import List, Optional

from app.db.sqlite_client import SQLiteClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _fmt_inr(v) -> str:
    if v is None:
        return "—"
    try:
        return f"₹{int(v):,}"
    except (TypeError, ValueError):
        return str(v)


class KBQueryService:
    def __init__(self, db: SQLiteClient):
        self.db = db

    def _rows(self, sql: str, params: tuple = ()) -> List:
        try:
            return self.db.fetchall(sql, params)
        except Exception as e:
            logger.debug("kb query failed (tables may not exist yet)", extra={"err": str(e)})
            return []

    @property
    def available(self) -> bool:
        return bool(self._rows("SELECT 1 FROM kb_meetings LIMIT 1"))

    # ── Meetings ───────────────────────────────────────────────────────
    def meetings_table(self) -> str:
        rows = self._rows("SELECT meeting_no, meeting_date, financial_year, source_docs FROM kb_meetings ORDER BY meeting_no")
        if not rows:
            return ""
        lines = ["| Meeting | Date | FY | Source |", "|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['meeting_no']} | {r['meeting_date'] or '—'} | {r['financial_year'] or '—'} | {r['source_docs']} |")
        return "### All CSR meetings (exact, from knowledge base)\n" + "\n".join(lines)

    def meeting_detail(self, meeting_no: int) -> str:
        out: List[str] = []
        m = self._rows("SELECT * FROM kb_meetings WHERE meeting_no=?", (meeting_no,))
        if m:
            r = m[0]
            out.append(f"### Meeting {meeting_no} (exact): date {r['meeting_date'] or 'unknown'}, FY {r['financial_year'] or 'unknown'}")
        att = self._rows(
            "SELECT person, designation, role FROM kb_attendance WHERE meeting_no=? ORDER BY role, person",
            (meeting_no,),
        )
        if att:
            present = [f"{a['person']} ({a['designation'] or a['role']})" for a in att if a["role"] != "absent"]
            absent = [a["person"] for a in att if a["role"] == "absent"]
            if present:
                out.append("Attendees: " + "; ".join(present))
            if absent:
                out.append("Leave of absence: " + "; ".join(absent))
        dec = self._rows(
            "SELECT agenda_item, title, decision_text, project_name, amount, source_doc FROM kb_decisions WHERE meeting_no=? ORDER BY agenda_item",
            (meeting_no,),
        )
        if dec:
            out.append("Decisions:")
            for d in dec:
                item = f"{d['agenda_item']} — " if d["agenda_item"] else ""
                amt = f" ({_fmt_inr(d['amount'])})" if d["amount"] else ""
                out.append(f"- {item}{d['title'] or ''}: {d['decision_text']}{amt} [src: {d['source_doc']}]")
        return "\n".join(out)

    # ── People / attendance ────────────────────────────────────────────
    def person_attendance(self, person_query: str) -> str:
        q = f"%{person_query.strip().lower()}%"
        rows = self._rows(
            "SELECT person, COUNT(DISTINCT meeting_no) AS n, GROUP_CONCAT(DISTINCT meeting_no) AS meetings "
            "FROM kb_attendance WHERE LOWER(person) LIKE ? AND role != 'absent' GROUP BY person",
            (q,),
        )
        if not rows:
            return ""
        out = ["### Attendance (exact, from knowledge base)"]
        for r in rows:
            out.append(f"- {r['person']}: attended {r['n']} meetings (meetings {r['meetings']})")
        return "\n".join(out)

    # ── Projects ───────────────────────────────────────────────────────
    def projects_table(self) -> str:
        rows = self._rows(
            "SELECT project_name, ngo_name, sector, status, approved_cost FROM kb_projects ORDER BY project_name"
        )
        if not rows:
            return ""
        lines = ["| Project | NGO | Sector | Status | Approved |", "|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['project_name'][:70]} | {r['ngo_name'] or '—'} | {r['sector'] or '—'} | {r['status']} | {_fmt_inr(r['approved_cost'])} |"
            )
        return "### Canonical project registry (exact, from knowledge base)\n" + "\n".join(lines)

    def project_card(self, name_query: str) -> str:
        q = f"%{name_query.strip().lower()}%"
        rows = self._rows(
            "SELECT * FROM kb_projects WHERE LOWER(project_name) LIKE ? OR LOWER(ngo_name) LIKE ? LIMIT 3",
            (q, q),
        )
        out: List[str] = []
        for r in rows:
            out.append(
                f"### Project (exact): {r['project_name']}\n"
                f"NGO: {r['ngo_name'] or '—'} | Sector: {r['sector'] or '—'} | Status: {r['status']} | "
                f"Approved: {_fmt_inr(r['approved_cost'])} | Disbursed: {_fmt_inr(r['disbursed_amount'])} | "
                f"Location: {r['district'] or r['state'] or '—'}\n"
                f"Sources: {r['source_docs']}"
            )
            events = self._rows(
                "SELECT meeting_no, stage, status, approved_cost, notes, source_doc FROM kb_project_events "
                "WHERE project_key=? ORDER BY COALESCE(meeting_no, 999)",
                (r["project_key"],),
            )
            if events:
                out.append("Lifecycle:")
                for e in events:
                    mtg = f"Meeting {e['meeting_no']}" if e["meeting_no"] else "(no meeting ref)"
                    amt = f", {_fmt_inr(e['approved_cost'])}" if e["approved_cost"] else ""
                    note = f" — {e['notes']}" if e["notes"] else ""
                    out.append(f"- {mtg}: {e['stage']} / {e['status']}{amt}{note} [src: {e['source_doc']}]")
        return "\n".join(out)

    def ngo_projects(self, ngo_query: str) -> str:
        q = f"%{ngo_query.strip().lower()}%"
        rows = self._rows(
            "SELECT project_name, status, approved_cost, source_docs FROM kb_projects WHERE LOWER(ngo_name) LIKE ?",
            (q,),
        )
        if not rows:
            return ""
        total = sum(r["approved_cost"] or 0 for r in rows)
        out = [f"### NGO projects (exact): matching '{ngo_query}'"]
        for r in rows:
            out.append(f"- {r['project_name']} — {r['status']} — {_fmt_inr(r['approved_cost'])} [src: {r['source_docs']}]")
        out.append(f"Total approved across these projects: {_fmt_inr(total)}")
        return "\n".join(out)

    # ── Budgets ────────────────────────────────────────────────────────
    def budget_lines(self, financial_year: Optional[str] = None, meeting_no: Optional[int] = None) -> str:
        sql = "SELECT financial_year, meeting_no, line_item, amount, kind, source_doc FROM kb_budget_lines WHERE 1=1"
        params: list = []
        if financial_year:
            sql += " AND financial_year LIKE ?"
            params.append(f"%{financial_year}%")
        if meeting_no is not None:
            sql += " AND meeting_no=?"
            params.append(meeting_no)
        sql += " ORDER BY financial_year, meeting_no LIMIT 60"
        rows = self._rows(sql, tuple(params))
        if not rows:
            return ""
        lines = ["| FY | Meeting | Line item | Amount | Kind | Source |", "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['financial_year'] or '—'} | {r['meeting_no'] or '—'} | {r['line_item'][:60]} | {_fmt_inr(r['amount'])} | {r['kind'] or '—'} | {r['source_doc']} |"
            )
        return "### Budget records (exact, from knowledge base)\n" + "\n".join(lines)

    # ── Router entry point ─────────────────────────────────────────────
    def blocks_for_plan(self, plan: dict, question: str) -> str:
        """Given the planner output, return the authoritative KB blocks for
        this query class. Empty string when nothing applies."""
        intent = plan.get("intent", "general_faq")
        ents = plan.get("entities", {}) or {}
        blocks: List[str] = []

        mnum = ents.get("meeting_number")
        try:
            mnum = int(mnum) if mnum is not None else None
        except (TypeError, ValueError):
            mnum = None

        # Meeting-class questions
        if mnum is not None:
            blocks.append(self.meeting_detail(mnum))
            blocks.append(self.budget_lines(meeting_no=mnum))
        if intent.startswith("meeting") and mnum is None:
            blocks.append(self.meetings_table())

        # Project-class questions
        proj = ents.get("project_name")
        if proj:
            blocks.append(self.project_card(proj))
        ngo = ents.get("ngo_name")
        if ngo:
            blocks.append(self.ngo_projects(ngo))
        if intent.startswith(("project", "timeline")) and not proj and not ngo:
            blocks.append(self.projects_table())

        # Budget-class questions
        if intent in ("budget_lookup", "expenditure_lookup", "allocation_lookup", "meeting_budget"):
            blocks.append(self.budget_lines(financial_year=ents.get("financial_year"), meeting_no=mnum))

        # Attendance questions. The planner extracts person_name + the
        # person_attendance intent; prefer those, then fall back to keyword +
        # capitalized-name detection so older/looser phrasings still resolve.
        ql = question.lower()
        person = (ents.get("person_name") or "").strip()
        is_attendance = (
            intent == "person_attendance"
            or bool(person)
            or any(w in ql for w in ("attend", "attendee", "present", "member", "who came", "chaired"))
        )
        if is_attendance and mnum is None:
            if person:
                # Surname-only lookup is the most robust (kb stores canonical full
                # names; "Rashmi Verma" and "Verma" both need to match).
                blocks.append(self.person_attendance(person))
                surname = person.split()[-1]
                if surname and surname.lower() != person.lower():
                    blocks.append(self.person_attendance(surname))
            else:
                names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", question)
                for n in names[:2]:
                    blocks.append(self.person_attendance(n))
                if not names:
                    blocks.append(self.meetings_table())

        # Corpus-wide aggregates
        if any(w in ql for w in ("all meeting", "every meeting", "list of meeting", "date wise", "datewise")):
            blocks.append(self.meetings_table())
        if any(w in ql for w in ("all project", "every project", "list of project", "how many project")):
            blocks.append(self.projects_table())

        merged = "\n\n".join(b for b in dict.fromkeys(blocks) if b)
        return merged
