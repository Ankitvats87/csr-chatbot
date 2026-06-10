"""V2 retrieval service — 4-phase strategy from csr.md.

Phase 1  Metadata Filtering  — extract structured filters from the question text
Phase 2  Vector Retrieval    — query csr_v2_enriched with filter + embedding
Phase 3  Lifecycle Assembly  — enrich results with csr_project_master records
Phase 4  Answer Generation   — unchanged; handled by ResponseService

V1 (knowledgebase namespace) is NEVER touched by this service.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from app.db.pinecone_client import PineconeClient
from app.ingestion_v2.pinecone_v2 import V2_NAMESPACE_ENRICHED, V2_NAMESPACE_PROJECT_MASTER
from app.models.message_model import RetrievedChunk
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Phase 1 helpers ──────────────────────────────────────────────────────────

_ORDINAL_RE = re.compile(r"\b(\d{1,3})(?:st|nd|rd|th)?\b", re.IGNORECASE)

_DOC_TYPE_MAP: Dict[str, List[str]] = {
    "CSR Agenda": ["agenda"],
    "CSR Minutes": ["minutes", "mom", "meeting minutes"],
    "Board Minutes": ["board minutes", "bod minutes", "board of directors minutes"],
    "Resolution by Circulation": ["resolution by circulation", "rbc", "by circulation"],
    "MOA": [r"\bmoa\b", "memorandum of agreement", "memorandum of association"],
    "Progress Report": ["progress report"],
    "Completion Report": ["completion report"],
}

_LIFECYCLE_MAP: Dict[str, List[str]] = {
    "Board_Approval": ["approved by board", "board approval", "board approved", "ratified by board"],
    "Committee_Recommendation": ["recommended by committee", "committee recommendation", "committee approved"],
    "Fund_Release": ["fund release", "disbursed", "installment", "payment released"],
    "MOA_Signed": ["moa signed", "moa executed", "memorandum signed"],
    "Progress_Update": ["progress update", "status update", "utilisation certificate"],
    "Completion": ["completed", "completion"],
    "Proposal": ["proposal", "proposed"],
    "Amendment": ["amendment", "modification", "revised"],
}


def _extract_meeting_numbers(question: str) -> List[int]:
    nums: List[int] = []
    for m in _ORDINAL_RE.finditer(question):
        try:
            n = int(m.group(1))
            if 1 <= n <= 999:
                nums.append(n)
        except ValueError:
            continue
    return nums


def _extract_doc_type(question: str) -> Optional[str]:
    ql = question.lower()
    for doc_type, keywords in _DOC_TYPE_MAP.items():
        for kw in keywords:
            if re.search(kw, ql):
                return doc_type
    return None


def _extract_lifecycle(question: str) -> Optional[str]:
    ql = question.lower()
    for stage, keywords in _LIFECYCLE_MAP.items():
        for kw in keywords:
            if kw in ql:
                return stage
    return None


# ── Service ──────────────────────────────────────────────────────────────────

class VectorServiceV2:
    """Drop-in replacement for VectorService that queries V2 namespaces."""

    # Top-K per namespace before threshold filtering.
    _TOP_K_ENRICHED = 12
    _TOP_K_MASTER = 5

    # V2 thresholds are set 0.10 below the global SIMILARITY_THRESHOLD because V2
    # chunks include "[Document: ...] [Section: ...] [Page N]" prefixes that
    # dilute cosine similarity by ~0.05–0.10 vs. raw V1 chunks. Project masters
    # are short prose summaries and score even lower; they get an extra -0.05.
    _ENRICHED_THRESHOLD_OFFSET = 0.10
    _MASTER_THRESHOLD_OFFSET = 0.15
    _MIN_THRESHOLD = 0.30

    def __init__(self, pinecone: PineconeClient, settings: Settings, hybrid=None):
        self.pinecone = pinecone
        self.settings = settings
        self.hybrid = hybrid  # Optional[HybridSearchService] — BM25 fusion when enabled
        self._enriched_threshold = max(
            settings.similarity_threshold - self._ENRICHED_THRESHOLD_OFFSET,
            self._MIN_THRESHOLD,
        )
        self._master_threshold = max(
            settings.similarity_threshold - self._MASTER_THRESHOLD_OFFSET,
            self._MIN_THRESHOLD,
        )

    # ── Public interface (same signature as VectorService) ────────────
    def query(
        self,
        embedding: List[float],
        question: str = "",
        metadata_filter: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        # Phase 1: build metadata filter from question if caller didn't supply one.
        if metadata_filter is None and question:
            metadata_filter = self._build_filter(question)

        # Phase 2: enriched chunk retrieval.
        enriched = self._query_namespace(
            embedding,
            namespace=V2_NAMESPACE_ENRICHED,
            top_k=top_k or self._TOP_K_ENRICHED,
            metadata_filter=metadata_filter,
        )

        # Fallback: if a metadata filter shrank the result to zero, retry without it.
        # Prevents over-strict filters (e.g. "MOA_Signed" when only 2 such chunks exist)
        # from killing otherwise-relevant semantic matches.
        if metadata_filter and not enriched:
            logger.info("v2 filter returned 0 — retrying without filter", extra={"dropped_filter": metadata_filter})
            enriched = self._query_namespace(
                embedding,
                namespace=V2_NAMESPACE_ENRICHED,
                top_k=top_k or self._TOP_K_ENRICHED,
                metadata_filter=None,
            )

        # Phase 3: project lifecycle assembly — always run, lowers threshold slightly
        # so we pull in the most relevant project master even for borderline matches.
        masters = self._query_project_masters(embedding)

        # Phase 2b: hybrid BM25 fusion — lexical recall for exact tokens
        # (amounts, names, ordinals) that cosine similarity under-ranks.
        if (
            question
            and self.settings.enable_hybrid_retrieval
            and self.hybrid is not None
            and self.hybrid.available
        ):
            lexical = self.hybrid.lexical_search(question)
            enriched = self.hybrid.fuse(enriched, lexical)[: self.settings.context_max_chunks]

        return self._merge(enriched, masters)

    def upsert(self, vectors: List[dict]) -> None:
        """No-op in V2 — ingestion goes through V2Pipeline/V2Pinecone."""
        pass

    def delete_by_file_id(self, file_id: str) -> int:
        """Delegates to V2Pinecone prefix-delete on csr_v2_enriched."""
        from app.ingestion_v2.pinecone_v2 import V2Pinecone
        v2 = V2Pinecone(self.pinecone)
        return v2.delete_chunks_by_file_id(file_id)

    # ── Phase 1: metadata filter builder ─────────────────────────────
    def _build_filter(self, question: str) -> Optional[dict]:
        conditions: List[dict] = []

        meeting_numbers = _extract_meeting_numbers(question)
        if meeting_numbers:
            # Pinecone stores meeting_number as a numeric value (float). Send int — Pinecone coerces.
            if len(meeting_numbers) == 1:
                conditions.append({"meeting_number": {"$eq": meeting_numbers[0]}})
            else:
                conditions.append({"meeting_number": {"$in": meeting_numbers}})

        doc_type = _extract_doc_type(question)
        if doc_type:
            conditions.append({"document_type": {"$eq": doc_type}})

        lifecycle = _extract_lifecycle(question)
        if lifecycle:
            conditions.append({"lifecycle_stage": {"$eq": lifecycle}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ── Phase 2: namespace query ──────────────────────────────────────
    def _query_namespace(
        self,
        embedding: List[float],
        namespace: str,
        top_k: int,
        metadata_filter: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        kwargs: dict = {
            "vector": embedding,
            "top_k": top_k,
            "namespace": namespace,
            "include_metadata": True,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter

        try:
            resp = self.pinecone.index.query(**kwargs)
        except Exception as e:
            logger.exception(
                "v2 pinecone query failed",
                extra={"namespace": namespace, "err": str(e)},
            )
            return []

        matches = resp.get("matches") if isinstance(resp, dict) else getattr(resp, "matches", [])
        chunks: List[RetrievedChunk] = []
        for m in matches or []:
            score = m["score"] if isinstance(m, dict) else m.score
            if score is None or score < self._enriched_threshold:
                continue
            md = (m["metadata"] if isinstance(m, dict) else (m.metadata or {})) or {}
            chunk_id = m["id"] if isinstance(m, dict) else m.id
            page = md.get("page")
            chunks.append(
                RetrievedChunk(
                    text=md.get("text", ""),
                    score=float(score),
                    document_name=md.get("document_name"),
                    source=namespace,
                    page=str(page) if page is not None else None,
                    chunk_id=chunk_id,
                )
            )

        logger.info(
            "v2 enriched query",
            extra={
                "namespace": namespace,
                "raw": len(matches or []),
                "kept": len(chunks),
                "threshold": self._enriched_threshold,
                "filter": bool(metadata_filter),
            },
        )
        return chunks

    # ── Phase 3: project lifecycle assembly ──────────────────────────
    def _query_project_masters(self, embedding: List[float]) -> List[RetrievedChunk]:
        master_threshold = self._master_threshold
        kwargs: dict = {
            "vector": embedding,
            "top_k": self._TOP_K_MASTER,
            "namespace": V2_NAMESPACE_PROJECT_MASTER,
            "include_metadata": True,
        }
        try:
            resp = self.pinecone.index.query(**kwargs)
        except Exception as e:
            logger.warning("v2 project master query failed", extra={"err": str(e)})
            return []

        from app.ingestion_v2.project_master_builder import is_generic_label

        matches = resp.get("matches") if isinstance(resp, dict) else getattr(resp, "matches", [])
        chunks: List[RetrievedChunk] = []
        for m in matches or []:
            score = m["score"] if isinstance(m, dict) else m.score
            if score is None or score < master_threshold:
                continue
            md = (m["metadata"] if isinstance(m, dict) else (m.metadata or {})) or {}
            project_name = md.get("project_name", "Unknown Project")
            # Stale letter-label masters ("Project C") may persist in Pinecone
            # until the next full ingestion — never surface them.
            if is_generic_label(project_name):
                continue
            summary = md.get("summary_text", "")
            chunk_id = m["id"] if isinstance(m, dict) else m.id
            chunks.append(
                RetrievedChunk(
                    text=f"[Project Master — {project_name}]\n{summary}",
                    score=float(score),
                    document_name=f"Project Master — {project_name}",
                    source=V2_NAMESPACE_PROJECT_MASTER,
                    page=None,
                    chunk_id=chunk_id,
                )
            )

        logger.info(
            "v2 project master query",
            extra={"raw": len(matches or []), "kept": len(chunks)},
        )
        return chunks

    # ── Merge & deduplicate ───────────────────────────────────────────
    @staticmethod
    def _merge(
        enriched: List[RetrievedChunk],
        masters: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        seen: Set[str] = set()
        out: List[RetrievedChunk] = []
        # Masters first — anchor the project lifecycle context at the top.
        for c in list(masters) + list(enriched):
            key = c.chunk_id or (c.document_name or "") + "::" + c.text[:40]
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out
