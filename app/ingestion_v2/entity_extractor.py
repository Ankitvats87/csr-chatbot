"""LLM-based structured entity extraction.

For each parsed document, we send the FULL markdown text (LlamaParse
output) to OpenAI gpt-4o-mini and request the ExtractedDocument schema
back via JSON-mode. The model never sees more than one document at a
time, so context overflows are unlikely on these CSR PDFs (~50-100 pages
each chunked-internally if needed).

Uses the openai SDK directly (NOT via OpenRouter) so we get reliable
structured output.
"""
from __future__ import annotations

import json
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.ingestion_v2.schemas import (
    Attendee,
    Beneficiary,
    BudgetLine,
    CSRClassification,
    Decision,
    DocumentType,
    ExtractedDocument,
    Financial,
    Geography,
    Governance,
    LifecycleStage,
    Meeting,
    NGO,
    Project,
    ProjectStatus,
)
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# NOTE: "filename" is a reserved Python logging LogRecord attribute — never use it
# as an extra key or logging raises KeyError.  We use "doc_name" instead everywhere.

_SYSTEM_PROMPT = """You are a precise information extraction system for Corporate Social Responsibility (CSR) committee documents from an Indian public-sector company.

You will receive ONE document (markdown form). Extract every CSR-relevant entity into a strict JSON object matching the schema described in the user message.

Hard rules — read carefully:
1. Extract ONLY what is EXPLICITLY stated in the document. Do not infer, guess, or fill placeholder data.
2. If a field is not mentioned, return null (or an empty list for list fields).
3. Numbers: convert Indian-format numbers to plain integers (e.g. "Rs. 1,15,34,300" → 11534300; "95,00,000" → 9500000).
4. Dates: convert to ISO YYYY-MM-DD when possible; otherwise return the date string as written.
5. Project names: copy from the document verbatim; collect any alternative phrasings into project_aliases.
6. NGO names: copy verbatim; do not abbreviate or rephrase.
7. Many CSR documents discuss MULTIPLE projects — list each as a separate Project entry in `projects`.
8. The same project may appear in multiple lifecycle stages within one document (proposal + budget + approval). Pick the FURTHEST stage discussed for that project in this document.
9. `references_to_prior_meetings` should list any ordinals like "as decided in the 24th CSR meeting" → [24].
10. `high_level_summary` is at most 3 lines, plain English, no markdown.
11. `lifecycle_stage` must be one of: Proposal, Committee_Recommendation, Board_Approval, MOA_Signed, Fund_Release, Amendment, Progress_Update, Completion, Unknown.
12. `project_status` must be one of: Proposed, Under_Consideration, Approved, MOA_Executed, In_Progress, Completed, Amended, Closed, Unknown.
13. `meeting` must always be a JSON object (not null). If meeting details are unavailable use {}.
14. `attendees`: every person recorded as present at (or granted leave of absence from) the meeting.
    Strip salutations (Shri/Smt./Dr./Mr./Ms.) from names. Keep designation as written.
    role = 'member' | 'invitee' | 'secretary' | 'absent'.
15. `decisions`: one entry per agenda item that was discussed/resolved/noted. Agenda items are numbered
    <meeting>.<item> (e.g. "26.3" = 26th meeting, item 3) — copy that number into agenda_item when present.
    decision_text records WHAT was decided (approved/noted/deferred/ratified), faithful to the document.
16. `budget_lines`: every concrete monetary figure with what it is for — annual action plan totals,
    per-project approvals, expenditures, balances, administrative overheads. Convert to plain INR numbers.
17. Per-meeting letter labels like "Project A" / "NGO B" are table shorthand, NOT names. If a table uses
    letters but the descriptive name appears elsewhere in the document, use the descriptive name.
    If no descriptive name exists anywhere, OMIT that row entirely — never output a letter label as a name.

If the document is empty or non-CSR (e.g. a cover letter), return document_type='Unknown', empty projects list, and a brief explanation in high_level_summary."""


_USER_TEMPLATE = """### Document filename
{filename}

### Heuristic document type (you may override)
{heuristic_type}

### Markdown content
<<<DOCUMENT>>>
{markdown}
<<<END>>>

Return STRICT JSON matching this Pydantic schema (no comments, no trailing keys):
{schema_json}
"""


class EntityExtractor:
    """Wraps a direct OpenAI client (not OpenRouter) so structured-output mode
    behaves as documented. Cost: ~$0.01 per document at gpt-4o-mini pricing.
    """

    MODEL = "gpt-4o-mini"
    MAX_INPUT_CHARS = 280_000  # ≈ 70k tokens — safely under gpt-4o-mini's 128k context

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY required for V2 entity extraction. Set it in .env."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._schema_json = ExtractedDocument.model_json_schema()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=12))
    def extract(
        self,
        *,
        filename: str,
        markdown: str,
        heuristic_type: DocumentType,
    ) -> ExtractedDocument:
        # Hard-truncate ridiculously long documents (rare on CSR PDFs).
        if len(markdown) > self.MAX_INPUT_CHARS:
            logger.warning(
                "document truncated for extraction",
                extra={"doc_name": filename, "original_chars": len(markdown), "kept_chars": self.MAX_INPUT_CHARS},
            )
            markdown = markdown[: self.MAX_INPUT_CHARS]

        user_msg = _USER_TEMPLATE.format(
            filename=filename,
            heuristic_type=heuristic_type.value,
            markdown=markdown,
            schema_json=json.dumps(self._schema_json, ensure_ascii=False)[:14000],
        )
        resp = self._client.chat.completions.create(
            model=self.MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.exception(
                "extractor returned invalid JSON",
                extra={"doc_name": filename, "err": str(e), "raw_preview": raw[:300]},
            )
            raise

        # Validate against pydantic — invalid fields are dropped/coerced.
        try:
            doc = ExtractedDocument.model_validate(parsed)
        except Exception as e:
            logger.warning(
                "schema validation soft-failed; using lenient parse",
                extra={"doc_name": filename, "err": str(e)[:200]},
            )
            doc = self._lenient_parse(parsed)

        logger.info(
            "extracted entities",
            extra={
                "doc_name": filename,
                "doc_type": doc.document_type.value,
                "n_projects": len(doc.projects),
                "meeting_number": doc.meeting.meeting_number,
            },
        )
        return doc

    @staticmethod
    def _lenient_parse(payload: dict) -> ExtractedDocument:
        """When strict validation fails, recover as many fields as possible so
        ingestion can continue. Each sub-section is wrapped individually so one
        bad field doesn't lose all the others.
        """
        safe = ExtractedDocument()

        # Document type
        try:
            dt = payload.get("document_type")
            if dt:
                valid = {v.value for v in DocumentType}
                safe.document_type = DocumentType(dt) if dt in valid else DocumentType.unknown
        except Exception:
            pass

        # Meeting — LLM sometimes returns null; default to empty Meeting object
        try:
            m = payload.get("meeting")
            if isinstance(m, dict):
                safe.meeting = Meeting(
                    meeting_number=m.get("meeting_number"),
                    meeting_date=m.get("meeting_date"),
                    financial_year=m.get("financial_year"),
                )
            # If m is None or not a dict, keep the default Meeting()
        except Exception:
            pass

        # Governance
        try:
            g = payload.get("governance")
            if isinstance(g, dict):
                safe.governance = Governance(
                    agenda_items=g.get("agenda_items") or [],
                    resolution_numbers=g.get("resolution_numbers") or [],
                    board_meeting_number=g.get("board_meeting_number"),
                    committee_name=g.get("committee_name"),
                )
        except Exception:
            pass

        # High-level summary
        try:
            safe.high_level_summary = payload.get("high_level_summary")
        except Exception:
            pass

        # References to prior meetings
        try:
            refs = payload.get("references_to_prior_meetings") or []
            safe.references_to_prior_meetings = [int(r) for r in refs if r is not None]
        except Exception:
            pass

        # Attendees / decisions / budget lines — each row individually
        for a in (payload.get("attendees") or []):
            try:
                if isinstance(a, dict) and a.get("name"):
                    safe.attendees.append(Attendee(
                        name=a["name"],
                        designation=a.get("designation"),
                        role=a.get("role"),
                    ))
            except Exception:
                pass
        for d in (payload.get("decisions") or []):
            try:
                if isinstance(d, dict) and d.get("decision_text"):
                    safe.decisions.append(Decision(
                        agenda_item=d.get("agenda_item"),
                        title=d.get("title"),
                        decision_text=d["decision_text"],
                        project_name=d.get("project_name"),
                        amount=d.get("amount"),
                    ))
            except Exception:
                pass
        for b in (payload.get("budget_lines") or []):
            try:
                if isinstance(b, dict) and b.get("line_item"):
                    safe.budget_lines.append(BudgetLine(
                        financial_year=b.get("financial_year"),
                        line_item=b["line_item"],
                        amount=b.get("amount"),
                        kind=b.get("kind"),
                    ))
            except Exception:
                pass

        # Projects — parse each individually so one bad project doesn't lose all
        _valid_stages = {v.value for v in LifecycleStage}
        _valid_statuses = {v.value for v in ProjectStatus}

        parsed_projects: list[Project] = []
        for p in (payload.get("projects") or []):
            if not isinstance(p, dict):
                continue
            try:
                ls_raw = p.get("lifecycle_stage", "Unknown")
                if ls_raw not in _valid_stages:
                    ls_raw = "Unknown"
                ps_raw = p.get("project_status", "Unknown")
                if ps_raw not in _valid_statuses:
                    ps_raw = "Unknown"

                proj = Project(
                    project_id=p.get("project_id"),
                    project_name=p.get("project_name") or "Unknown Project",
                    project_aliases=p.get("project_aliases") or [],
                    project_status=ProjectStatus(ps_raw),
                    lifecycle_stage=LifecycleStage(ls_raw),
                    notes=p.get("notes"),
                )

                try:
                    ngo_raw = p.get("ngo")
                    if isinstance(ngo_raw, dict) and ngo_raw.get("ngo_name"):
                        proj.ngo = NGO(
                            ngo_name=ngo_raw["ngo_name"],
                            ngo_type=ngo_raw.get("ngo_type"),
                            registration_date=ngo_raw.get("registration_date"),
                        )
                except Exception:
                    pass

                try:
                    fin_raw = p.get("financial")
                    if isinstance(fin_raw, dict):
                        proj.financial = Financial(
                            project_cost=fin_raw.get("project_cost"),
                            approved_cost=fin_raw.get("approved_cost"),
                            disbursed_amount=fin_raw.get("disbursed_amount"),
                            balance_amount=fin_raw.get("balance_amount"),
                        )
                except Exception:
                    pass

                try:
                    geo_raw = p.get("geography")
                    if isinstance(geo_raw, dict):
                        proj.geography = Geography(
                            state=geo_raw.get("state"),
                            district=geo_raw.get("district"),
                            city=geo_raw.get("city"),
                        )
                except Exception:
                    pass

                try:
                    ben_raw = p.get("beneficiary")
                    if isinstance(ben_raw, dict):
                        proj.beneficiary = Beneficiary(
                            beneficiary_count=ben_raw.get("beneficiary_count"),
                            beneficiary_type=ben_raw.get("beneficiary_type"),
                        )
                except Exception:
                    pass

                try:
                    cls_raw = p.get("classification")
                    if isinstance(cls_raw, dict):
                        proj.classification = CSRClassification(
                            csr_schedule=cls_raw.get("csr_schedule"),
                            csr_sector=cls_raw.get("csr_sector"),
                        )
                except Exception:
                    pass

                parsed_projects.append(proj)
            except Exception:
                pass

        safe.projects = parsed_projects
        return safe
