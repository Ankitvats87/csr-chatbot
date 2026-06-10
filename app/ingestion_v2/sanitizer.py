"""Parsed-markdown sanitizer + validation gate.

LlamaParse's LLM mode, when given a system prompt that DESCRIBES expected
content, fabricates example tables for pages it cannot read ("Project A |
1,000,000 | NGO A", attendees "John Doe / Jane Smith"). We fixed the prompt,
but defense-in-depth demands that NOTHING resembling placeholder content can
enter the index regardless of what the parser returns.

Two functions:

  sanitize_markdown(md)  -> (clean_md, removed_blocks)
      Strips table rows / lines that match known fabrication signatures and
      drops tables left with no real data rows.

  validate_markdown(md)  -> list of issue strings
      Run AFTER sanitization: any surviving fabrication marker is reported,
      and the pipeline refuses to index the document (fail-closed).
"""
from __future__ import annotations

import re
from typing import List, Tuple

# Signatures of fabricated placeholder content. Deliberately narrow — these
# patterns never occur in genuine PTC CSR documents.
_PLACEHOLDER_ROW_RE = re.compile(
    r"^\|\s*(?:project|ngo)\s+[a-z]{1,2}\d{0,2}\s*\|", re.IGNORECASE
)
_PLACEHOLDER_NAMES_RE = re.compile(
    r"\b(john\s+doe|jane\s+smith|robert\s+brown|emily\s+white|lorem\s+ipsum)\b",
    re.IGNORECASE,
)
_TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _is_fabricated_line(line: str) -> bool:
    return bool(_PLACEHOLDER_ROW_RE.match(line.strip())) or bool(
        _PLACEHOLDER_NAMES_RE.search(line)
    )


def sanitize_markdown(md: str) -> Tuple[str, List[str]]:
    """Remove fabricated rows/lines. Returns (clean_markdown, removed_lines)."""
    if not md:
        return md, []

    lines = md.splitlines()
    keep: List[str] = []
    removed: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_LINE_RE.match(line):
            # Collect the whole contiguous table block.
            block = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                block.append(lines[i])
                i += 1
            data_rows = [
                l for l in block
                if not _TABLE_SEPARATOR_RE.match(l)
            ]
            fabricated = [l for l in data_rows if _is_fabricated_line(l)]
            genuine_data = [
                l for l in data_rows[1:]  # exclude header row
                if not _is_fabricated_line(l)
            ]
            if fabricated and not genuine_data:
                # Entire table is placeholder content — drop it whole.
                removed.extend(block)
            elif fabricated:
                # Mixed table — drop only the fabricated rows.
                for l in block:
                    if _is_fabricated_line(l):
                        removed.append(l)
                    else:
                        keep.append(l)
            else:
                keep.extend(block)
            continue

        if _is_fabricated_line(line):
            removed.append(line)
        else:
            keep.append(line)
        i += 1

    return "\n".join(keep), removed


def validate_markdown(md: str) -> List[str]:
    """Post-sanitization gate. Any hit here means the document must NOT be
    indexed until inspected — fail closed, never index suspect content."""
    issues: List[str] = []
    for n, line in enumerate(md.splitlines(), 1):
        if _is_fabricated_line(line):
            issues.append(f"line {n}: fabrication marker survived: {line.strip()[:90]}")
    return issues
