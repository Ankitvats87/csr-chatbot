"""Aggregates per-document Project entries into one ProjectMaster per
canonical project across the entire corpus.

Canonicalization rule: project names are compared after lower-casing,
removing punctuation, and dropping common stopwords ("project", "scheme",
"initiative", "csr"). Aliases declared in any document also fold into
the same master.

Project ID minting: if a Project entry already has a project_id from the
source document we keep it; otherwise we mint one as `CSR-<FY>-<seq>`
where FY is the financial year of the EARLIEST meeting that mentioned
the project and seq is a 3-digit counter within that FY.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.ingestion_v2.schemas import (
    ExtractedDocument,
    LifecycleStage,
    Project,
    ProjectMaster,
    ProjectStatus,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


_STOPWORDS = {"project", "projects", "scheme", "initiative", "csr", "the", "and", "of", "a", "for"}

# Source agendas label proposals per-meeting as "Project A", "Project B", …
# (and sometimes "NGO A"). These letters are NOT global project identities —
# Project A of meeting 26 has nothing to do with Project A of meeting 23 —
# so they must never become canonical Project Masters.
_GENERIC_NAME_RE = re.compile(r"^\s*(project|ngo)\s*[-:]?\s*[a-z]{1,2}\d{0,2}\s*$", re.IGNORECASE)


def is_generic_label(name: Optional[str]) -> bool:
    return bool(name) and bool(_GENERIC_NAME_RE.match(name))


def _canon_key(name: str) -> str:
    """Aggressive canonical key for matching project names across docs."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s).lower()
    tokens = [t for t in s.split() if t and t not in _STOPWORDS]
    return " ".join(sorted(tokens))


@dataclass
class _Accumulator:
    canon_key: str
    primary_name: str = ""
    aliases: List[str] = field(default_factory=list)
    ngo_names: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    schedule_clauses: List[str] = field(default_factory=list)
    statuses: List[ProjectStatus] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    districts: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    approved_costs: List[float] = field(default_factory=list)
    disbursed_amounts: List[float] = field(default_factory=list)
    balance_amounts: List[float] = field(default_factory=list)
    beneficiary_counts: List[int] = field(default_factory=list)
    beneficiary_types: List[str] = field(default_factory=list)
    lifecycle_stages_crossed: List[LifecycleStage] = field(default_factory=list)
    meeting_numbers: List[int] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    earliest_fy: Optional[str] = None
    source_project_id: Optional[str] = None  # if any doc gave one


class ProjectMasterBuilder:
    def __init__(self):
        self._acc: Dict[str, _Accumulator] = {}

    def add_document(self, doc: ExtractedDocument, source_filename: str) -> None:
        for proj in doc.projects:
            self._absorb_project(proj, doc, source_filename)

    def _absorb_project(self, proj: Project, doc: ExtractedDocument, source: str) -> None:
        # Drop per-meeting letter labels ("Project A", "NGO B") — they are
        # table shorthand inside one agenda, not real project identities.
        if is_generic_label(proj.project_name):
            logger.info(
                "skipping generic letter-label project",
                extra={"name": proj.project_name, "source": source},
            )
            return
        # Match the project to an existing accumulator via canonical name OR alias.
        candidate_names = [proj.project_name] + list(proj.project_aliases or [])
        keys = [_canon_key(n) for n in candidate_names if n]
        keys = [k for k in keys if k]
        if not keys:
            return

        # Use the FIRST non-empty key as the merge target.
        target_key = keys[0]
        # If any candidate key already exists, fold into that one.
        for k in keys:
            if k in self._acc:
                target_key = k
                break

        acc = self._acc.setdefault(target_key, _Accumulator(canon_key=target_key))
        if not acc.primary_name:
            acc.primary_name = proj.project_name
        for n in candidate_names:
            if n and n != acc.primary_name and n not in acc.aliases:
                acc.aliases.append(n)

        # Source bookkeeping
        if source not in acc.source_documents:
            acc.source_documents.append(source)
        if doc.meeting.meeting_number is not None and doc.meeting.meeting_number not in acc.meeting_numbers:
            acc.meeting_numbers.append(doc.meeting.meeting_number)
        if proj.lifecycle_stage and proj.lifecycle_stage != LifecycleStage.unknown:
            if proj.lifecycle_stage not in acc.lifecycle_stages_crossed:
                acc.lifecycle_stages_crossed.append(proj.lifecycle_stage)

        # NGO + sector + schedule
        if proj.ngo and proj.ngo.ngo_name:
            if proj.ngo.ngo_name not in acc.ngo_names:
                acc.ngo_names.append(proj.ngo.ngo_name)
        if proj.classification:
            if proj.classification.csr_sector and proj.classification.csr_sector not in acc.sectors:
                acc.sectors.append(proj.classification.csr_sector)
            if proj.classification.csr_schedule and proj.classification.csr_schedule not in acc.schedule_clauses:
                acc.schedule_clauses.append(proj.classification.csr_schedule)

        # Geography
        if proj.geography:
            if proj.geography.state and proj.geography.state not in acc.states:
                acc.states.append(proj.geography.state)
            if proj.geography.district and proj.geography.district not in acc.districts:
                acc.districts.append(proj.geography.district)
            if proj.geography.city and proj.geography.city not in acc.cities:
                acc.cities.append(proj.geography.city)

        # Financial — keep the maximum approved cost seen and most-recent disbursed/balance
        if proj.financial:
            if proj.financial.approved_cost is not None:
                acc.approved_costs.append(proj.financial.approved_cost)
            if proj.financial.disbursed_amount is not None:
                acc.disbursed_amounts.append(proj.financial.disbursed_amount)
            if proj.financial.balance_amount is not None:
                acc.balance_amounts.append(proj.financial.balance_amount)

        # Beneficiary
        if proj.beneficiary:
            if proj.beneficiary.beneficiary_count is not None:
                acc.beneficiary_counts.append(proj.beneficiary.beneficiary_count)
            if proj.beneficiary.beneficiary_type and proj.beneficiary.beneficiary_type not in acc.beneficiary_types:
                acc.beneficiary_types.append(proj.beneficiary.beneficiary_type)

        # Status: keep the FURTHEST status seen (lifecycle order)
        if proj.project_status and proj.project_status != ProjectStatus.unknown:
            acc.statuses.append(proj.project_status)

        # Earliest FY
        if doc.meeting.financial_year:
            if acc.earliest_fy is None or doc.meeting.financial_year < acc.earliest_fy:
                acc.earliest_fy = doc.meeting.financial_year

        # Source-given project_id (rare but use if present)
        if proj.project_id and not acc.source_project_id:
            acc.source_project_id = proj.project_id

    # ──────────────────────────────────────────────────────────────────
    # Mint masters
    # ──────────────────────────────────────────────────────────────────
    def build(self) -> List[ProjectMaster]:
        results: List[ProjectMaster] = []
        # Stable, deterministic ID allocation per FY.
        per_fy_counter: Dict[str, int] = {}

        # Sort accumulators by earliest_fy ascending then by primary_name.
        ordered = sorted(
            self._acc.values(),
            key=lambda a: ((a.earliest_fy or "9999-99"), a.primary_name.lower()),
        )
        for acc in ordered:
            if acc.source_project_id:
                pid = acc.source_project_id
            else:
                fy = acc.earliest_fy or "UNK"
                per_fy_counter[fy] = per_fy_counter.get(fy, 0) + 1
                short_fy = fy.replace("-", "_") if fy != "UNK" else "UNK"
                pid = f"CSR-{short_fy}-{per_fy_counter[fy]:03d}"

            status = self._pick_furthest_status(acc.statuses)
            master = ProjectMaster(
                project_id=pid,
                project_name=acc.primary_name,
                aliases=acc.aliases,
                ngo_name=acc.ngo_names[0] if acc.ngo_names else None,
                sector=acc.sectors[0] if acc.sectors else None,
                schedule_vii_clause=acc.schedule_clauses[0] if acc.schedule_clauses else None,
                current_status=status,
                geography=self._fold_geography(acc),
                approved_cost=max(acc.approved_costs) if acc.approved_costs else None,
                disbursed_amount=acc.disbursed_amounts[-1] if acc.disbursed_amounts else None,
                balance_amount=acc.balance_amounts[-1] if acc.balance_amounts else None,
                beneficiary_count=max(acc.beneficiary_counts) if acc.beneficiary_counts else None,
                beneficiary_type=acc.beneficiary_types[0] if acc.beneficiary_types else None,
                lifecycle_stages_crossed=acc.lifecycle_stages_crossed,
                meeting_numbers_referenced=sorted(acc.meeting_numbers),
                source_documents=acc.source_documents,
                summary_text="",  # filled below
            )
            master.summary_text = self._synthesize_summary(master)
            results.append(master)

        logger.info("project masters built", extra={"n_masters": len(results)})
        return results

    @staticmethod
    def _pick_furthest_status(statuses: List[ProjectStatus]) -> ProjectStatus:
        order = [
            ProjectStatus.closed,
            ProjectStatus.completed,
            ProjectStatus.in_progress,
            ProjectStatus.moa_executed,
            ProjectStatus.amended,
            ProjectStatus.approved,
            ProjectStatus.under_consideration,
            ProjectStatus.proposed,
        ]
        for s in order:
            if s in statuses:
                return s
        return ProjectStatus.unknown

    @staticmethod
    def _fold_geography(acc: _Accumulator):
        from app.ingestion_v2.schemas import Geography
        if not (acc.states or acc.districts or acc.cities):
            return None
        return Geography(
            state=acc.states[0] if acc.states else None,
            district=acc.districts[0] if acc.districts else None,
            city=acc.cities[0] if acc.cities else None,
        )

    @staticmethod
    def _synthesize_summary(m: ProjectMaster) -> str:
        bits = [f"Project: {m.project_name} ({m.project_id})."]
        if m.aliases:
            bits.append(f"Also known as: {', '.join(m.aliases)}.")
        if m.ngo_name:
            bits.append(f"Implementing NGO: {m.ngo_name}.")
        if m.sector:
            bits.append(f"Sector: {m.sector}.")
        if m.schedule_vii_clause:
            bits.append(f"Schedule VII clause: {m.schedule_vii_clause}.")
        if m.geography:
            g_parts = [x for x in [m.geography.city, m.geography.district, m.geography.state] if x]
            if g_parts:
                bits.append(f"Location: {', '.join(g_parts)}.")
        if m.approved_cost is not None:
            bits.append(f"Approved cost: INR {int(m.approved_cost):,}.")
        if m.disbursed_amount is not None:
            bits.append(f"Disbursed to date: INR {int(m.disbursed_amount):,}.")
        if m.balance_amount is not None:
            bits.append(f"Balance: INR {int(m.balance_amount):,}.")
        if m.beneficiary_count is not None:
            bits.append(f"Beneficiaries: {m.beneficiary_count:,} {m.beneficiary_type or ''}.".strip())
        if m.current_status != ProjectStatus.unknown:
            bits.append(f"Current status: {m.current_status.value}.")
        if m.lifecycle_stages_crossed:
            stages = ", ".join(s.value for s in m.lifecycle_stages_crossed)
            bits.append(f"Lifecycle stages reached: {stages}.")
        if m.meeting_numbers_referenced:
            mnums = ", ".join(str(n) for n in m.meeting_numbers_referenced)
            bits.append(f"Discussed in CSR meetings: {mnums}.")
        if m.source_documents:
            bits.append(f"Source documents: {'; '.join(m.source_documents[:6])}.")
        return " ".join(bits)
