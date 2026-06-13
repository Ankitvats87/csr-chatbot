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
_HTML_TAG_RE = re.compile(r"</?(?:b|i|code|pre)>")
_NUMERIC_RE = re.compile(r"^-?\d+(?:[\d,]*\.\d+)?$")

PRE_TABLE_WIDTH_LIMIT = 60
PRE_TABLE_COL_MAX = 26


def _is_numeric_like(cell: str) -> bool:
    s = re.sub(r"[₹$€£¥%,\s]", "", cell)
    return bool(s and _NUMERIC_RE.match(s))


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"


def _abbreviate_header(h: str) -> str:
    # CSR docs verbose headers waste 2/3 of the row width; keep a useful short form.
    h_stripped = re.sub(r"\([^)]*\)", "", h).strip()
    if " / " in h_stripped:
        h_stripped = h_stripped.split(" / ", 1)[0].strip()
    if "/" in h_stripped and len(h_stripped) > 14:
        h_stripped = h_stripped.split("/", 1)[0].strip()
    return _truncate(h_stripped, PRE_TABLE_COL_MAX)


_TOTAL_KEYWORDS = {"total", "totals", "grand total", "sub total", "subtotal"}


def _render_pre_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a GitHub-style markdown pipe table inside Telegram <pre>...</pre>.

    Format:
        | #  | Entity            | Amount |
        |----|-------------------|--------|
        |  1 | PTC Vishram Sadan | 600.00 |
        |    | TOTAL             | 999.00 |

    Numeric columns are right-aligned, text columns left-aligned, long cells
    truncated with … so the row stays mobile-friendly. A separator row is
    inserted above any TOTAL row to visually anchor it."""
    n = len(headers)
    norm_rows = [r + [""] * (n - len(r)) for r in rows]

    is_num = [
        all(
            (not r[i]) or r[i] in ("—", "-") or _is_numeric_like(r[i])
            for r in norm_rows
        )
        for i in range(n)
    ]
    abbr_headers = [_abbreviate_header(h) for h in headers]

    widths = [
        min(
            max(len(abbr_headers[i]), *(len(r[i]) for r in norm_rows)) if norm_rows else len(abbr_headers[i]),
            PRE_TABLE_COL_MAX,
        )
        for i in range(n)
    ]

    def fmt(cells: List[str]) -> str:
        parts: List[str] = []
        for i, c in enumerate(cells[:n]):
            c = _truncate(c, widths[i])
            parts.append(c.rjust(widths[i]) if is_num[i] else c.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"

    body_lines: List[str] = [fmt(abbr_headers), sep]
    for r in norm_rows:
        first = (r[0] or "").strip().lower()
        if first in _TOTAL_KEYWORDS or (not r[0] and any(s in (r[i] or "").strip().lower() for i in range(1, n) for s in _TOTAL_KEYWORDS)):
            body_lines.append(sep)
        body_lines.append(fmt(r))

    return "\n<pre>\n" + "\n".join(body_lines) + "\n</pre>\n"


def _table_fits_pre(headers: List[str], rows: List[List[str]]) -> bool:
    """A pipe table fits the <pre> width budget when sum(col widths) plus the
    pipe/space overhead stays under PRE_TABLE_WIDTH_LIMIT. Overhead per row is
    `| ` + ` | ` × (n-1) + ` |` = 3*(n-1) + 4 chars."""
    n = len(headers)
    cols_w: List[int] = []
    for i in range(n):
        col_w = len(_abbreviate_header(headers[i]))
        for r in rows:
            v = r[i] if i < len(r) else ""
            col_w = max(col_w, min(len(v), PRE_TABLE_COL_MAX))
        cols_w.append(col_w)
    overhead = 3 * (n - 1) + 4
    total = sum(cols_w) + overhead
    return total <= PRE_TABLE_WIDTH_LIMIT


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
            rows: List[List[str]] = []
            while i < len(lines) and lines[i].count("|") >= 2 and _TABLE_ROW_RE.match(lines[i]):
                cells = _split_table_row(lines[i])
                cells += [""] * (len(headers) - len(cells))
                rows.append(cells)
                i += 1
            if _table_fits_pre(headers, rows):
                out.append(_render_pre_table(headers, rows))
            else:
                records: List[str] = []
                for cells in rows:
                    title = cells[0] or "(item)"
                    block = [f"**{title}**"]
                    for h, v in zip(headers[1:], cells[1 : len(headers)]):
                        if v:
                            block.append(f"{h}: {v}" if h else v)
                    records.append("\n".join(block))
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
    in_pre = False
    for raw in text.splitlines():
        line = raw.rstrip()

        if line == "<pre>":
            in_pre = True
            lines.append(line)
            continue
        if line == "</pre>":
            in_pre = False
            lines.append(line)
            continue
        if in_pre:
            # Inside a <pre> block: escape HTML but preserve all whitespace
            # and skip every other transform — the table alignment depends on it.
            lines.append(html.escape(line, quote=False))
            continue

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
    in_pre = False
    for raw in text.splitlines():
        line = raw.rstrip()

        if line == "<pre>" or line == "</pre>":
            in_pre = (line == "<pre>")
            continue
        if in_pre:
            # Already aligned by _render_pre_table — pass through as-is.
            lines.append(line)
            continue

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
