"""Hybrid lexical + vector retrieval (Enhance Retrieval spec).

Dense embeddings miss exact tokens — budget figures ("11534300"), resolution
numbers, NGO names, meeting ordinals. NotebookLM never loses those because it
grounds on full source text. This module closes that gap:

  1. ChunkStore   — local JSON mirror of every csr_v2_enriched vector's
                    metadata (id, text, document_name, page, entities).
                    Built by scripts/build_chunk_store.py or kept in sync
                    automatically by the V2 ingestion pipeline.
  2. BM25Index    — pure-Python Okapi BM25 over the chunk store. No new
                    dependencies, fast enough for a corpus of this size
                    (hundreds–low thousands of chunks).
  3. RRF fusion   — Reciprocal Rank Fusion merges the vector ranking and the
                    BM25 ranking into one list; a chunk found by BOTH rises.
  4. Entity boost — chunks whose metadata matches entities extracted by the
                    query planner (meeting number, project, NGO) get a
                    multiplicative score boost.

Everything degrades gracefully: if the chunk store file is missing the
service reports unavailable and callers fall back to pure vector retrieval.
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from app.models.message_model import RetrievedChunk
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# RRF constant — standard value from the original Cormack et al. paper.
RRF_K = 60


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Okapi BM25 (k1=1.5, b=0.75), pure Python."""

    K1 = 1.5
    B = 0.75

    def __init__(self, tokenized_docs: Sequence[List[str]]):
        self.n_docs = len(tokenized_docs)
        self.doc_freqs: List[Counter] = [Counter(toks) for toks in tokenized_docs]
        self.doc_lens = [len(toks) for toks in tokenized_docs]
        self.avg_len = (sum(self.doc_lens) / self.n_docs) if self.n_docs else 0.0
        # document frequency per term
        df: Counter = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        # idf with the standard BM25 floor at 0 (avoid negative idf for very common terms)
        self.idf: Dict[str, float] = {
            term: max(0.0, math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0))
            for term, n in df.items()
        }

    def search(self, query: str, top_k: int) -> List[tuple[int, float]]:
        """Returns [(doc_index, score)] sorted by score desc, scores > 0 only."""
        q_tokens = tokenize(query)
        if not q_tokens or self.n_docs == 0:
            return []
        scores = [0.0] * self.n_docs
        for term in q_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(term)
                if not tf:
                    continue
                denom = tf + self.K1 * (1 - self.B + self.B * self.doc_lens[i] / self.avg_len)
                scores[i] += idf * tf * (self.K1 + 1) / denom
        ranked = [(i, s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class HybridSearchService:
    """Loads the local chunk store, exposes lexical search + RRF fusion + boost."""

    def __init__(self, settings: Settings, project_root: Optional[Path] = None):
        self.settings = settings
        root = project_root or Path(__file__).resolve().parent.parent.parent
        p = Path(settings.chunk_store_path)
        self.store_path = p if p.is_absolute() else root / p
        self._lock = threading.Lock()
        self.chunks: List[dict] = []
        self.index: Optional[BM25Index] = None
        self._meta_by_id: Dict[str, dict] = {}
        self.reload()

    # ── store loading ──────────────────────────────────────────────────
    @property
    def available(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def reload(self) -> None:
        """(Re)load the chunk store from disk and rebuild the BM25 index."""
        with self._lock:
            if not self.store_path.is_file():
                logger.warning(
                    "chunk store not found — hybrid lexical search disabled "
                    "(run scripts/build_chunk_store.py to enable)",
                    extra={"path": str(self.store_path)},
                )
                self.chunks, self.index, self._meta_by_id = [], None, {}
                return
            try:
                payload = json.loads(self.store_path.read_text(encoding="utf-8"))
                self.chunks = payload.get("chunks", [])
                tokenized = [tokenize(self._searchable_text(c)) for c in self.chunks]
                self.index = BM25Index(tokenized)
                self._meta_by_id = {c["id"]: c.get("metadata", {}) for c in self.chunks}
                logger.info(
                    "hybrid chunk store loaded",
                    extra={"n_chunks": len(self.chunks), "path": str(self.store_path)},
                )
            except Exception as e:
                logger.exception("failed to load chunk store", extra={"err": str(e)})
                self.chunks, self.index, self._meta_by_id = [], None, {}

    @staticmethod
    def _searchable_text(chunk: dict) -> str:
        """Text + identifying metadata so 'meeting 26' / NGO names match lexically."""
        md = chunk.get("metadata", {})
        parts = [
            str(md.get("document_name", "")),
            str(md.get("section_path", "")),
            str(md.get("document_type", "")),
        ]
        mnum = md.get("meeting_number")
        if mnum not in (None, ""):
            # emit both "26" and "meeting 26" forms
            n = str(int(mnum)) if isinstance(mnum, float) else str(mnum)
            parts.append(f"meeting {n}")
        for key in ("project_names", "ngo_names", "csr_sectors", "states"):
            v = md.get(key)
            if isinstance(v, list):
                parts.extend(str(x) for x in v)
        parts.append(chunk.get("text", ""))
        return " ".join(parts)

    # ── lexical search ─────────────────────────────────────────────────
    def lexical_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        if not self.available:
            return []
        k = top_k or self.settings.hybrid_bm25_top_k
        hits = self.index.search(query, k)
        out: List[RetrievedChunk] = []
        for idx, score in hits:
            c = self.chunks[idx]
            md = c.get("metadata", {})
            page = md.get("page")
            out.append(
                RetrievedChunk(
                    text=c.get("text", "") or md.get("text", ""),
                    score=float(score),
                    document_name=md.get("document_name"),
                    source="csr_v2_enriched",
                    page=str(int(page)) if isinstance(page, float) else (str(page) if page not in (None, "") else None),
                    chunk_id=c["id"],
                )
            )
        logger.info("bm25 lexical search", extra={"query": query[:60], "n_hits": len(out)})
        return out

    # ── fusion ─────────────────────────────────────────────────────────
    @staticmethod
    def fuse(
        vector_chunks: List[RetrievedChunk],
        lexical_chunks: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Reciprocal Rank Fusion. A chunk's fused score is the sum of
        1/(RRF_K + rank) over every list it appears in, so agreement between
        dense and lexical retrieval dominates either signal alone."""

        def key(c: RetrievedChunk) -> str:
            return c.chunk_id or f"{c.document_name}::{(c.text or '')[:40]}"

        fused: Dict[str, RetrievedChunk] = {}
        scores: Dict[str, float] = {}
        for ranked in (vector_chunks, lexical_chunks):
            for rank, c in enumerate(ranked):
                k = key(c)
                scores[k] = scores.get(k, 0.0) + 1.0 / (RRF_K + rank + 1)
                # keep the variant with the longest text (Pinecone + store both carry it)
                if k not in fused or len(c.text or "") > len(fused[k].text or ""):
                    fused[k] = c
        out: List[RetrievedChunk] = []
        for k, c in fused.items():
            c.score = scores[k]
            out.append(c)
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    # ── entity boost ───────────────────────────────────────────────────
    def boost_by_entities(
        self,
        chunks: List[RetrievedChunk],
        meeting_numbers: Optional[Set[int]] = None,
        project_names: Optional[Set[str]] = None,
        ngo_names: Optional[Set[str]] = None,
    ) -> List[RetrievedChunk]:
        """Multiplicative boost per matched entity dimension using the chunk
        store's metadata. Chunks not in the store pass through unchanged."""
        if not (meeting_numbers or project_names or ngo_names):
            return chunks
        proj_l = {p.lower() for p in (project_names or set())}
        ngo_l = {n.lower() for n in (ngo_names or set())}
        for c in chunks:
            md = self._meta_by_id.get(c.chunk_id or "")
            if not md:
                continue
            boost = 1.0
            mnum = md.get("meeting_number")
            if meeting_numbers and mnum not in (None, ""):
                try:
                    if int(float(mnum)) in meeting_numbers:
                        boost *= 1.3
                except (TypeError, ValueError):
                    pass
            if proj_l:
                chunk_projects = {str(p).lower() for p in md.get("project_names", []) or []}
                if any(p in cp or cp in p for p in proj_l for cp in chunk_projects):
                    boost *= 1.25
            if ngo_l:
                chunk_ngos = {str(n).lower() for n in md.get("ngo_names", []) or []}
                if any(n in cn or cn in n for n in ngo_l for cn in chunk_ngos):
                    boost *= 1.25
            c.score *= boost
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
