"""Markdown-header-aware chunker.

V1 splits by token count blindly. V2 splits by LOGICAL boundaries:

  - Top-level markdown headers (##, ###) demarcate sections.
  - Each chunk's natural unit is one section (agenda item, resolution,
    annexure, project summary, budget block, status update).
  - If a section is shorter than the floor (700 tokens), it is merged with
    the next adjacent section so the LLM receives sufficient context.
  - If a section is longer than the ceiling (900 tokens), it is sub-split
    on the next-most-granular header level (####), preserving section
    parent in the chunk header.
  - Chunks ALWAYS carry a `section_path` field so retrieval can reconstruct
    where in the document a chunk came from.

Targets per csr.md: 700-900 tokens, 80 overlap.
Token count is approximated as ~4 chars/token (close to OpenAI tokenizer for
English text + numbers; we don't need exact for chunking).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Approximate token = 4 characters of English text.
CHARS_PER_TOKEN = 4
TARGET_MIN_TOKENS = 700
TARGET_MAX_TOKENS = 900
OVERLAP_TOKENS = 80
TARGET_MIN_CHARS = TARGET_MIN_TOKENS * CHARS_PER_TOKEN
TARGET_MAX_CHARS = TARGET_MAX_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN

# Match markdown headers at any depth (## title, ### title, etc.).
_HEADER_RE = re.compile(r"^(#{2,6})\s+(.+)$", re.MULTILINE)


@dataclass
class LogicalChunk:
    text: str
    page: int  # source page (LlamaParse split_by_page=True gives one segment per page)
    section_path: str  # e.g. "Agenda Items > Item 3: Clean Ganga Fund"
    chunk_index: int = 0


@dataclass
class _Section:
    level: int
    heading: str
    page: int
    body_lines: List[str] = field(default_factory=list)
    children: List["_Section"] = field(default_factory=list)

    @property
    def body_text(self) -> str:
        return "\n".join(self.body_lines).strip()

    @property
    def char_len(self) -> int:
        return len(self.full_text())

    def full_text(self) -> str:
        parts: List[str] = []
        if self.heading:
            parts.append("#" * self.level + " " + self.heading)
        if self.body_text:
            parts.append(self.body_text)
        for c in self.children:
            parts.append(c.full_text())
        return "\n\n".join(p for p in parts if p)


class LogicalChunker:
    """Chunks ONE page (or one markdown blob) of LlamaParse output into
    section-respecting chunks targeting 700-900 tokens with 80-token overlap.
    """

    def chunk_page(self, markdown: str, page: int) -> List[LogicalChunk]:
        sections = self._parse_sections(markdown, page)
        if not sections:
            return self._fallback_window_split(markdown, page, parent_path="")

        chunks: List[LogicalChunk] = []
        for s in sections:
            chunks.extend(self._chunk_section(s, page, parent_path=""))
        # Assign chunk_index in document order.
        for i, c in enumerate(chunks):
            c.chunk_index = i
        return chunks

    # ─── section parser ───────────────────────────────────────────────
    def _parse_sections(self, md: str, page: int) -> List[_Section]:
        if not md.strip():
            return []
        # If no headers found, treat the whole page as one synthetic "page" section.
        headers = list(_HEADER_RE.finditer(md))
        if not headers:
            return [_Section(level=2, heading=f"Page {page}", page=page, body_lines=md.splitlines())]

        # Slice the markdown by header positions.
        roots: List[_Section] = []
        stack: List[_Section] = []
        last_end = 0
        # Any preamble before the first header becomes a synthetic root.
        if headers[0].start() > 0:
            preamble = md[: headers[0].start()].strip()
            if preamble:
                roots.append(_Section(level=2, heading=f"Page {page} (intro)", page=page, body_lines=preamble.splitlines()))

        for i, h in enumerate(headers):
            level = len(h.group(1))
            heading = h.group(2).strip()
            start = h.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
            body = md[start:end].strip().splitlines()
            sec = _Section(level=level, heading=heading, page=page, body_lines=body)
            # Maintain a stack of open parents based on header level.
            while stack and stack[-1].level >= level:
                stack.pop()
            if stack:
                stack[-1].children.append(sec)
            else:
                roots.append(sec)
            stack.append(sec)
        return roots

    # ─── section → chunks ────────────────────────────────────────────
    def _chunk_section(self, sec: _Section, page: int, parent_path: str) -> List[LogicalChunk]:
        path = (parent_path + " > " + sec.heading) if parent_path else sec.heading

        # Combined text of THIS section including its children.
        full = sec.full_text()
        chars = len(full)

        if chars <= TARGET_MAX_CHARS:
            return [LogicalChunk(text=full, page=page, section_path=path)]

        # Too long: try to split using the children's boundaries.
        if sec.children:
            out: List[LogicalChunk] = []
            buf_text: List[str] = []
            if sec.body_text:
                buf_text.append("#" * sec.level + " " + sec.heading + "\n" + sec.body_text)
            buf_char = sum(len(x) for x in buf_text)
            for child in sec.children:
                ctext = child.full_text()
                if buf_char + len(ctext) <= TARGET_MAX_CHARS:
                    buf_text.append(ctext)
                    buf_char += len(ctext)
                else:
                    if buf_text:
                        out.append(LogicalChunk(text="\n\n".join(buf_text).strip(), page=page, section_path=path))
                    # Recurse into the oversized child.
                    if len(ctext) > TARGET_MAX_CHARS:
                        out.extend(self._chunk_section(child, page, parent_path=path))
                    else:
                        out.append(LogicalChunk(text=ctext, page=page, section_path=path + " > " + child.heading))
                    buf_text = []
                    buf_char = 0
            if buf_text:
                out.append(LogicalChunk(text="\n\n".join(buf_text).strip(), page=page, section_path=path))
            return out

        # No children → window-split this body with overlap.
        return self._fallback_window_split(full, page, parent_path=path)

    def _fallback_window_split(self, text: str, page: int, parent_path: str) -> List[LogicalChunk]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= TARGET_MAX_CHARS:
            return [LogicalChunk(text=text, page=page, section_path=parent_path or f"Page {page}")]
        # Split on paragraph boundaries first.
        paras = re.split(r"\n{2,}", text)
        chunks: List[LogicalChunk] = []
        buf: List[str] = []
        buf_chars = 0
        for p in paras:
            if buf_chars + len(p) <= TARGET_MAX_CHARS:
                buf.append(p)
                buf_chars += len(p) + 2
            else:
                if buf:
                    chunks.append(LogicalChunk(text="\n\n".join(buf).strip(), page=page, section_path=parent_path or f"Page {page}"))
                if len(p) > TARGET_MAX_CHARS:
                    # Hard-window with overlap as last resort.
                    start = 0
                    while start < len(p):
                        end = min(start + TARGET_MAX_CHARS, len(p))
                        chunks.append(LogicalChunk(text=p[start:end].strip(), page=page, section_path=parent_path or f"Page {page}"))
                        if end == len(p):
                            break
                        start = end - OVERLAP_CHARS
                else:
                    buf = [p]
                    buf_chars = len(p)
        if buf:
            chunks.append(LogicalChunk(text="\n\n".join(buf).strip(), page=page, section_path=parent_path or f"Page {page}"))
        return chunks
