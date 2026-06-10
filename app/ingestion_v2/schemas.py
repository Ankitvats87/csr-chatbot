"""Pydantic schemas for V2 entity extraction.

These define the EXACT structured output the LLM must produce for each
document. The schema mirrors the entity taxonomy in csr.md.

The V2 pipeline NEVER imports these from V1 code paths; V1 remains
completely unchanged.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums — closed vocabularies for classification fields
# ─────────────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    csr_agenda = "CSR Agenda"
    csr_minutes = "CSR Minutes"
    board_minutes = "Board Minutes"
    resolution_by_circulation = "Resolution by Circulation"
    moa = "MOA"
    progress_report = "Progress Report"
    completion_report = "Completion Report"
    unknown = "Unknown"


class LifecycleStage(str, Enum):
    proposal = "Proposal"
    committee_recommendation = "Committee_Recommendation"
    board_approval = "Board_Approval"
    moa_signed = "MOA_Signed"
    fund_release = "Fund_Release"
    amendment = "Amendment"
    progress_update = "Progress_Update"
    completion = "Completion"
    unknown = "Unknown"


class ProjectStatus(str, Enum):
    proposed = "Proposed"
    under_consideration = "Under_Consideration"
    approved = "Approved"
    moa_executed = "MOA_Executed"
    in_progress = "In_Progress"
    completed = "Completed"
    amended = "Amended"
    closed = "Closed"
    unknown = "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-entities
# ─────────────────────────────────────────────────────────────────────────────

class Meeting(BaseModel):
    meeting_number: Optional[int] = Field(None, description="Ordinal of the CSR meeting (e.g. 26 for 26th).")
    meeting_date: Optional[str] = Field(None, description="ISO date YYYY-MM-DD if discoverable; else None.")
    financial_year: Optional[str] = Field(None, description="Indian FY format e.g. '2024-25'.")


class Governance(BaseModel):
    agenda_items: List[str] = Field(default_factory=list, description="High-level agenda items in this document.")
    resolution_numbers: List[str] = Field(default_factory=list, description="Resolution / Circulation numbers cited.")
    board_meeting_number: Optional[str] = None
    committee_name: Optional[str] = Field(None, description="e.g. 'CSR Committee', 'Board of Directors'.")


class NGO(BaseModel):
    ngo_name: str
    ngo_type: Optional[str] = Field(None, description="e.g. 'Section 8 Company', 'Trust', 'Society'.")
    registration_date: Optional[str] = None


class Financial(BaseModel):
    project_cost: Optional[float] = Field(None, description="Total cost in INR.")
    approved_cost: Optional[float] = Field(None, description="CSR-approved amount in INR.")
    disbursed_amount: Optional[float] = None
    balance_amount: Optional[float] = None


class Geography(BaseModel):
    state: Optional[str] = None
    district: Optional[str] = None
    city: Optional[str] = None


class Beneficiary(BaseModel):
    beneficiary_count: Optional[int] = None
    beneficiary_type: Optional[str] = Field(None, description="e.g. 'Children', 'Women', 'Farmers', 'Patients'.")


class CSRClassification(BaseModel):
    csr_schedule: Optional[str] = Field(None, description="Schedule VII clause (e.g. 'i', 'ii', 'iii').")
    csr_sector: Optional[str] = Field(None, description="e.g. 'Healthcare', 'Education', 'Environment'.")


class Attendee(BaseModel):
    name: str = Field(description="Person's name exactly as written, minus salutations (drop Shri/Smt/Dr/Mr/Ms).")
    designation: Optional[str] = Field(None, description="e.g. 'Chairperson', 'Independent Director', 'CMD'.")
    role: Optional[str] = Field(None, description="'member' | 'invitee' | 'secretary' | 'absent' (leave-of-absence).")


class Decision(BaseModel):
    agenda_item: Optional[str] = Field(None, description="Agenda item number as written, e.g. '26.3' (= meeting 26, item 3).")
    title: Optional[str] = Field(None, description="Short title of the agenda item.")
    decision_text: str = Field(description="What was decided/resolved/noted, 1-3 lines, verbatim-faithful.")
    project_name: Optional[str] = Field(None, description="Descriptive project name if the decision concerns one project.")
    amount: Optional[float] = Field(None, description="INR amount tied to this decision, if any.")


class BudgetLine(BaseModel):
    financial_year: Optional[str] = Field(None, description="e.g. '2024-25'.")
    line_item: str = Field(description="What the amount is for (project, sector, 'Annual Action Plan total', 'Admin overhead', etc.)")
    amount: Optional[float] = Field(None, description="INR amount as a plain number.")
    kind: Optional[str] = Field(None, description="'allocation' | 'expenditure' | 'approved' | 'balance' | 'total'.")


class Project(BaseModel):
    project_id: Optional[str] = Field(None, description="If the doc gives one (e.g. CSR-2025-002); else leave blank — we mint one later.")
    project_name: str
    project_aliases: List[str] = Field(default_factory=list, description="Alternate names mentioned in the doc.")
    project_status: ProjectStatus = ProjectStatus.unknown
    ngo: Optional[NGO] = None
    financial: Optional[Financial] = None
    geography: Optional[Geography] = None
    beneficiary: Optional[Beneficiary] = None
    classification: Optional[CSRClassification] = None
    lifecycle_stage: LifecycleStage = LifecycleStage.unknown
    notes: Optional[str] = Field(None, description="Anything noteworthy that doesn't fit other fields (1-2 lines).")


# ─────────────────────────────────────────────────────────────────────────────
# Top-level document extraction
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedDocument(BaseModel):
    """The structured form of one parsed source document."""
    document_type: DocumentType = DocumentType.unknown
    meeting: Meeting = Field(default_factory=Meeting)
    governance: Governance = Field(default_factory=Governance)
    projects: List[Project] = Field(default_factory=list)
    attendees: List[Attendee] = Field(
        default_factory=list,
        description="People recorded as present (or on leave of absence) at the meeting.",
    )
    decisions: List[Decision] = Field(
        default_factory=list,
        description="One entry per agenda item that was decided/resolved/noted.",
    )
    budget_lines: List[BudgetLine] = Field(
        default_factory=list,
        description="Every budget/allocation/expenditure figure stated in the document.",
    )
    references_to_prior_meetings: List[int] = Field(
        default_factory=list,
        description="Meeting numbers cross-referenced in this document.",
    )
    high_level_summary: Optional[str] = Field(
        None, description="2-3 line plain-English summary of what this document is about."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Project Master — aggregated across all documents
# ─────────────────────────────────────────────────────────────────────────────

class ProjectMaster(BaseModel):
    project_id: str
    project_name: str
    aliases: List[str] = Field(default_factory=list)
    ngo_name: Optional[str] = None
    sector: Optional[str] = None
    schedule_vii_clause: Optional[str] = None
    current_status: ProjectStatus = ProjectStatus.unknown
    geography: Optional[Geography] = None
    approved_cost: Optional[float] = None
    disbursed_amount: Optional[float] = None
    balance_amount: Optional[float] = None
    beneficiary_count: Optional[int] = None
    beneficiary_type: Optional[str] = None
    lifecycle_stages_crossed: List[LifecycleStage] = Field(default_factory=list)
    meeting_numbers_referenced: List[int] = Field(default_factory=list)
    source_documents: List[str] = Field(default_factory=list)
    summary_text: str = Field(
        default="",
        description="Synthesized prose summary used as the embedded payload for csr_project_master.",
    )
