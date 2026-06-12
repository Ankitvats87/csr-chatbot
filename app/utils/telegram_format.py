"""Convert LLM markdown output into clean Telegram messages.

The LLM emits markdown (headings, bold, bullets, tables, inline [Ref N]
citations). Telegram's default mode renders none of that, so users see
literal `#`, `**` and broken `| pipe |` tables.

Two output flavours:
- format_for_telegram_html(): HTML for Telegram parse_mode=HTML — real
  <b>bold</b> headings and labels, bullets as •, tables re-flowed into
  per-record blocks (WhatsApp/Meta-AI-style cards).
- clean_for_telegram(): plain-text fallback with all markup stripped,
  used when Telegram rejects the HTML payload.
"""

import html
import re
from typing import List

_INLINE_REF_RE = re.compile(r"\s*\[Ref\s+\d+(?:\s*,\s*\d+)*\]")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_HRULE_RE = re.compile(r"^\s*[-=*_]{3,}\s*$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+")
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
_TABLE_ROW_RE = re.compile(r"^\s*\|?.*\|.*\|?\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_HTML_TAG_RE = re.compile(r"</?(?:b|i|code)>")


def _split_table_row(line: str) -> List[str]:
    # Cells lose their own emphasis markers — the record title gets bolded
    # uniformly downstream, and nested ** inside ** breaks the HTML pass.
    return [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]


def _convert_tables(text: str) -> str:
    """Re-flow markdown tables into per-record blocks.

    | Meeting | Agenda Date | Meeting Date |        **21st Meeting**
    |---------|-------------|--------------|   →    Agenda Date: 12 Jan 2024
    | 21st    | 12 Jan 2024 | 19 Jan 2024  |        Meeting Date: 19 Jan 2024

    The first column becomes the record's bold title; remaining columns
    become "Header: value" lines. Markdown `**` markers are emitted so the
    downstream bold handling (HTML or strip) applies uniformly.
    """
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # A table starts with a row containing >=2 pipes followed by a
        # separator row (|---|---|).
        if (
            line.count("|") >= 2
            and _TABLE_ROW_RE.match(line)
            and i + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[i + 1])
        ):
            headers = _split_table_row(line)
            i += 2
            records: List[str] = []
            while i < len(lines) and lines[i].count("|") >= 2 and _TABLE_ROW_RE.match(lines[i]):
                cells = _split_table_row(lines[i])
                cells += [""] * (len(headers) - len(cells))
                title = cells[0] or "(item)"
                block = [f"**{title}**"]
                for h, v in zip(headers[1:], cells[1 : len(headers)]):
                    if v:
                        block.append(f"{h}: {v}" if h else v)
                records.append("\n".join(block))
                i += 1
            out.append("\n\n".join(records))
            continue
        # Rogue single pipe-row without separator (model half-built a table):
        # render its cells on one line separated by " — ".
        if line.count("|") >= 2 and not _TABLE_SEP_RE.match(line):
            cells = [c for c in _split_table_row(line) if c]
            if len(cells) >= 2 and not any("|" in c for c in cells):
                joined = " — ".join(cells)
                # Heading-ish rows (short, no digits-only cells) keep bold.
                out.append(joined)
                i += 1
                continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _tidy(line: str) -> str:
    line = _INLINE_REF_RE.sub("", line)
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)
    line = re.sub(r"[ \t]{2,}", " ", line)
    return line.rstrip()


def format_for_telegram_html(text: str) -> str:
    """Return Telegram parse_mode=HTML text: bold headings/labels, • bullets,
    tables re-flowed, [Ref N] stripped. Tags never span lines, so splitting
    long messages on newlines stays valid HTML."""
    if not text:
        return ""

    text = _convert_tables(text)
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()

        if _HRULE_RE.match(line):
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            content = html.escape(_tidy(_BOLD_RE.sub(r"\1", heading.group(1).strip())))
            lines.append("")
            lines.append(f"<b>{content}</b>")
            continue

        bullet = _BULLET_RE.match(line)
        if bullet:
            line = f"{bullet.group(1)}• {line[bullet.end():]}"

        line = _tidy(line)
        line = html.escape(line, quote=False)
        # Escaping is done; now safe to add tags (escaped text has no <,>).
        line = _BOLD_RE.sub(r"<b>\1</b>", line)
        line = _ITALIC_RE.sub(r"<i>\1</i>", line)
        line = _BACKTICK_RE.sub(r"<code>\1</code>", line)
        if re.match(r"^\s*Sources:\s*$", line):
            line = "<b>Sources:</b>"

        lines.append(line)

    out = "\n".join(lines)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip()


def html_to_plain(text: str) -> str:
    """Strip the tags format_for_telegram_html added — plain-text fallback
    when Telegram rejects an HTML payload."""
    return html.unescape(_HTML_TAG_RE.sub("", text))


def clean_for_telegram(text: str) -> str:
    """Return Telegram-friendly plain text — no markdown markers, tidy bullets,
    tables re-flowed into record blocks, inline [Ref N] citations stripped."""
    if not text:
        return ""

    text = _convert_tables(text)
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

        lines.append(_tidy(line))

    out = "\n".join(lines)
    out = _MULTI_BLANK_RE.sub("\n\n", out)
    return out.strip()
