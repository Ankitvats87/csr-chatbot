import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timed() -> Iterator[dict]:
    """Usage: with timed() as t: ...  then t['ms']."""
    state: dict = {"ms": 0}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["ms"] = int((time.perf_counter() - start) * 1000)
