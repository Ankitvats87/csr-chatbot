from typing import List


def constant_time_equals(a: str, b: str) -> bool:
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= ord(x) ^ ord(y)
    return result == 0


def truncate(text: str, limit: int = 4000) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 20] + "\n…[truncated]"


def split_telegram_message(text: str, hard_limit: int = 4000) -> List[str]:
    """Telegram caps a single message at 4096 chars. Split on paragraph boundaries when possible.
    Never cuts inside a <pre> block — that would break the HTML and make the table render as raw text."""
    if not text:
        return [""]
    if len(text) <= hard_limit:
        return [text]
    parts: List[str] = []
    remaining = text
    while len(remaining) > hard_limit:
        chunk = remaining[:hard_limit]
        if chunk.count("<pre>") > chunk.count("</pre>"):
            # Would cut inside a <pre> block — back up to just before it opens.
            pre_start = chunk.rfind("<pre>")
            cut = pre_start if pre_start > hard_limit // 4 else hard_limit
        else:
            cut = remaining.rfind("\n\n", 0, hard_limit)
            if cut == -1:
                cut = remaining.rfind("\n", 0, hard_limit)
            if cut == -1 or cut < hard_limit // 2:
                cut = hard_limit
        parts.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        parts.append(remaining)
    return parts
