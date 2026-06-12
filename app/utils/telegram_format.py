"""Convert LLM markdown output into clean plain text for Telegram.

The LLM emits markdown (headings, bold, bullets, inline [Ref N] citations).
Telegram's default mode renders none of that, so users see literal `#` and
`**` clutter. We strip the noise but keep structure (headings on their own
line, bullets as •, blank lines preserved).
"""

import re
from typing import List


_INLINE_REF_RE = re.compile(r"\s*\[Ref\s+\d+(?:\s*,\s*\d+)*\]")
_SOURCES_HEADER_RE = re.compile(r"^\s*sources?\s*:?\s*$", re.IGNORECASE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_HRULE_RE = re.compile(r"^\s*[-=*_]{3,}\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")


def clean_for_telegram(text: str) -> str:
    """Return Telegram-friendly plain text — no markdown markers, tidy bullets,
    inline [Ref N] citations stripped (keeps a Sources block at the end if the
    LLM wrote one)."""
    if not text:
        return ""

    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()

        if _HRULE_RE.match(line):
            continue

        m = _HEADING_RE.match(line)
        if m:
            lines.append("")
            lines.append(m.group(1).strip())
            continue

        mb = _BULLET_RE.match(line)
        if mb:
            line = f"{mb.group(1)}• {line[mb.end():]}"

        line = _BOLD_RE.sub(r"\1", line)
        line = _ITALIC_RE.sub(r"\1", line)
        line = _BACKTICK_RE.sub(r"\1", line)

        line = _INLINE_REF_RE.sub("", line)

        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"[ \t]{2,}", " ", line)

        lines.append(line.rstrip())

    out = "\n".join(lines)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip()
