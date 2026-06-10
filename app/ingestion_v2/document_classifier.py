"""Heuristic classifier: maps filename + first-page text to a DocumentType.

Cheap and deterministic — used to PRE-LABEL before sending to the LLM.
The LLM may override during extraction, but starting with a confident
filename-based label massively improves extraction quality.
"""
from __future__ import annotations

import re
from typing import Optional

from app.ingestion_v2.schemas import DocumentType


def classify(filename: str, first_page_text: Optional[str] = None) -> DocumentType:
    n = filename.lower()
    t = (first_page_text or "").lower()

    # Most specific patterns first.
    if "completion report" in n or "completion report" in t:
        return DocumentType.completion_report
    if "progress report" in n or "progress report" in t:
        return DocumentType.progress_report
    if re.search(r"\bmoa\b|memorandum of association|memorandum of agreement", n + " " + t):
        return DocumentType.moa
    if "resolution by circulation" in n or "circulation no" in n or "circulation no" in t or "resolution by circulation" in t:
        return DocumentType.resolution_by_circulation
    if re.search(r"\bbod\b|board minutes|board of directors", n + " " + t):
        # Disambiguate: BOD minutes are board-level
        if "minutes" in n or "minutes" in t:
            return DocumentType.board_minutes
    if "csr" in n and "minutes" in n:
        return DocumentType.csr_minutes
    if "csr" in n and "agenda" in n:
        return DocumentType.csr_agenda

    # Fall back on first-page heuristics.
    if "csr committee" in t and "agenda" in t:
        return DocumentType.csr_agenda
    if "csr committee" in t and "minutes" in t:
        return DocumentType.csr_minutes
    if "board of directors" in t and "minutes" in t:
        return DocumentType.board_minutes

    return DocumentType.unknown
