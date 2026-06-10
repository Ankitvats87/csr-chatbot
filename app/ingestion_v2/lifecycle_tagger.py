"""Tags each chunk with the most likely lifecycle stage.

Uses high-signal keywords + simple priority rules. The csr.md schema
requires `lifecycle_stage` on every chunk; this keeps it deterministic
(no LLM cost per chunk) while staying interpretable.

If a chunk matches multiple stages, the one further along the project
lifecycle wins (Completion > Progress > Fund_Release > MOA > Board > Committee > Proposal),
because later-stage language usually summarizes earlier stages.
"""
from __future__ import annotations

import re
from typing import Tuple

from app.ingestion_v2.schemas import LifecycleStage

# Order matters: stages are listed from latest to earliest.
# First match wins (since later stages have priority).
_STAGE_PATTERNS: list[Tuple[LifecycleStage, list[str]]] = [
    (LifecycleStage.completion, [
        r"\bcompletion report\b",
        r"\bproject (?:has been )?completed\b",
        r"\bclos(?:ed|ure)\b",
        r"\bfinal(?:ised|ized)\b",
    ]),
    (LifecycleStage.progress_update, [
        r"\bprogress report\b",
        r"\bstatus update\b",
        r"\bimplementation status\b",
        r"\butilisation (?:report|certificate)\b",
        r"\bUC\b(?!\w)",
    ]),
    (LifecycleStage.amendment, [
        r"\bamendment\b",
        r"\bmodifi(?:cation|ed)\b",
        r"\brevised (?:scope|cost|moa|terms)\b",
        r"\bre[\-\s]?allocat\w*\b",
    ]),
    (LifecycleStage.fund_release, [
        r"\bfund(?:s)? (?:release|disburs)\w*\b",
        r"\bdisburs\w+\b",
        r"\bpay(?:ment|out|able) of (?:Rs\.|INR|₹)\b",
        r"\brelease(?:d)? to .{0,40} (?:NGO|implementing agency)\b",
        r"\binstallment (?:released|paid|disbursed)\b",
    ]),
    (LifecycleStage.moa_signed, [
        r"\bMOA (?:signed|executed|inked|entered into)\b",
        r"\bmemorandum of (?:agreement|association|understanding) (?:has been )?(?:signed|executed)\b",
        r"\bMoA executed\b",
    ]),
    (LifecycleStage.board_approval, [
        r"\b(?:approved|ratified|sanctioned) by (?:the )?board\b",
        r"\bboard (?:approval|approved|ratification|ratified)\b",
        r"\bresolution by circulation\b",
    ]),
    (LifecycleStage.committee_recommendation, [
        r"\bCSR committee (?:recommend|recommended|recommends)\b",
        r"\brecommend(?:ed|s|ation) to (?:the )?board\b",
        r"\bcommittee (?:approved|approval)\b",
    ]),
    (LifecycleStage.proposal, [
        r"\bproposal\b",
        r"\bproposed (?:project|scheme|initiative)\b",
        r"\bnew (?:project|proposal)\b",
        r"\bfor (?:consideration|approval) of (?:the )?committee\b",
        r"\bproject brief\b",
    ]),
]

_COMPILED = [
    (stage, [re.compile(p, re.IGNORECASE) for p in patterns])
    for stage, patterns in _STAGE_PATTERNS
]


def tag_chunk(text: str) -> LifecycleStage:
    """Return the most relevant LifecycleStage for this chunk text."""
    if not text:
        return LifecycleStage.unknown
    for stage, patterns in _COMPILED:
        for rgx in patterns:
            if rgx.search(text):
                return stage
    return LifecycleStage.unknown
