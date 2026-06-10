"""Re-use V1's LlamaParse extractor — but ALWAYS request markdown with
split_by_page=True (matches csr.md requirement of layout-aware parsing).

We keep the V1 chunker.py file untouched and import only `extract_segments`
through a thin re-export here so V2 has its own surface area.
"""
from __future__ import annotations

from typing import List

from app.ingestion.chunker import TextSegment, extract_segments  # re-use V1 parser path
from app.ingestion_v2.sanitizer import sanitize_markdown
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def parse_to_markdown(local_path: str, settings: Settings) -> List[TextSegment]:
    """Returns one TextSegment per page (markdown), via LlamaParse if
    LLAMA_CLOUD_API_KEY is set, otherwise falls back to PyPDF/python-docx
    just like V1. Every page is sanitized: fabricated placeholder content
    (parser hallucinations like 'Project A'/'John Doe' tables) is stripped
    before anything downstream can see it.
    """
    segments = extract_segments(local_path, settings)
    clean: List[TextSegment] = []
    total_removed = 0
    for seg in segments:
        text, removed = sanitize_markdown(seg.text)
        total_removed += len(removed)
        if text.strip():
            clean.append(TextSegment(text=text, page=seg.page))
    if total_removed:
        logger.warning(
            "sanitizer stripped fabricated parser content",
            extra={"path": local_path, "lines_removed": total_removed},
        )
    return clean


def joined_markdown(segments: List[TextSegment]) -> str:
    """Stitch per-page segments into a single markdown blob for the LLM
    entity extractor. Pages are separated by a clear delimiter so the
    model can still see page boundaries.
    """
    parts: List[str] = []
    for s in segments:
        parts.append(f"\n\n<!-- page {s.page} -->\n\n{s.text.strip()}")
    return "".join(parts).strip()
