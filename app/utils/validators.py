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
    """Telegram caps a single message at 4096 chars. Split on paragraph boundaries when possible."""
    if not text:
        return [""]
    if len(text) <= hard_limit:
        return [text]
    parts: List[str] = []
    remaining = text
    while len(remaining) > hard_limit:
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
