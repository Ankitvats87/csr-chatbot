"""Local-only retrieval evaluator (no Pinecone, no OpenAI).

Reuses the 100-question benchmark from evaluate_v2.py and runs each question
through the local BM25 chunk store + entity-aware boost. This gives a fast,
offline signal for whether ingestion/cleaning changes are moving accuracy.

Metrics per question:
  - hit            : at least one expected keyword found in top-K retrieved text
  - recall         : fraction of expected keywords found
  - n_chunks       : returned chunk count
  - top_score      : top BM25 + boost score
Aggregated as Hit Rate, Avg Recall, Avg Chunks.

Usage:
    python scripts/local_eval.py
    python scripts/local_eval.py --top-k 20 --out audit_reports/local_baseline.json
    python scripts/local_eval.py --tag after_cleanup
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.hybrid_search import BM25Index, HybridSearchService, tokenize
from app.utils.env import get_settings

# Reuse the canonical 100-question benchmark — keep evaluators comparable.
from evaluate_v2 import BENCHMARK  # noqa: E402


_MEETING_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?\b", re.IGNORECASE)


def extract_meeting_numbers(q: str) -> set[int]:
    out = set()
    for m in _MEETING_RE.finditer(q):
        try:
            n = int(m.group(1))
            if 5 <= n <= 99:  # meeting numbers in our corpus are 20-30
                out.add(n)
        except ValueError:
            pass
    return out


@dataclass
class QResult:
    question: str
    expected: List[str]
    matched: List[str] = field(default_factory=list)
    recall: float = 0.0
    hit: bool = False
    n_chunks: int = 0
    top_score: float = 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--chunk-store", type=str, default="", help="Override chunk store path")
    args = parser.parse_args()

    settings = get_settings()
    if args.chunk_store:
        settings = settings.model_copy(update={"chunk_store_path": args.chunk_store})
    svc = HybridSearchService(settings)
    if not svc.available:
        print("chunk store missing or empty; run scripts/build_chunk_store.py first", file=sys.stderr)
        return 1
    print(f"chunk store loaded: {len(svc.chunks)} chunks at {svc.store_path}")

    results: List[QResult] = []
    for q, kws in BENCHMARK:
        chunks = svc.lexical_search(q, top_k=args.top_k)
        meetings = extract_meeting_numbers(q)
        chunks = svc.boost_by_entities(chunks, meeting_numbers=meetings or None)
        combined = " ".join((c.text or "") for c in chunks).lower()
        matched = [kw for kw in kws if kw.lower() in combined]
        results.append(
            QResult(
                question=q,
                expected=kws,
                matched=matched,
                recall=len(matched) / len(kws) if kws else 0.0,
                hit=bool(matched),
                n_chunks=len(chunks),
                top_score=max((c.score for c in chunks), default=0.0),
            )
        )

    n = len(results)
    hit_rate = sum(r.hit for r in results) / n
    avg_recall = sum(r.recall for r in results) / n
    avg_chunks = sum(r.n_chunks for r in results) / n
    avg_score = sum(r.top_score for r in results) / n

    print()
    print(f"{'='*60}")
    print(f"  Local BM25 + entity boost  ({args.tag or 'untagged'})")
    print(f"  Questions : {n}")
    print(f"  Hit Rate  : {hit_rate:.1%}")
    print(f"  Avg Recall: {avg_recall:.1%}")
    print(f"  Avg Chunks: {avg_chunks:.1f}")
    print(f"  Avg Score : {avg_score:.3f}")
    print(f"{'='*60}")
    misses = [r for r in results if not r.hit]
    print(f"\n  Misses ({len(misses)}):")
    for r in misses[:25]:
        print(f"    - {r.question[:70]}   expected={r.expected}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "tag": args.tag,
                    "top_k": args.top_k,
                    "hit_rate": round(hit_rate, 4),
                    "avg_recall": round(avg_recall, 4),
                    "avg_chunks": round(avg_chunks, 2),
                    "avg_score": round(avg_score, 4),
                    "results": [asdict(r) for r in results],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
