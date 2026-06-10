"""V2 Pinecone wrapper.

Uses the SAME index as V1 (text-embedding-3-small → 1536 dim, cosine,
serverless aws/us-east-1) but writes to TWO new namespaces, leaving V1's
`knowledgebase` untouched:

  - csr_v2_enriched     → one vector per logical chunk (with rich metadata)
  - csr_project_master  → one vector per canonical project

The class is a thin facade over PineconeClient; we never touch the V1
index config or namespaces.
"""
from __future__ import annotations

from typing import Iterable, List

from app.db.pinecone_client import PineconeClient
from app.utils.logger import get_logger

logger = get_logger(__name__)

V2_NAMESPACE_ENRICHED = "csr_v2_enriched"
V2_NAMESPACE_PROJECT_MASTER = "csr_project_master"

UPSERT_BATCH = 100


class V2Pinecone:
    def __init__(self, pinecone_client: PineconeClient):
        self.pc = pinecone_client

    # ───── upsert ─────
    def upsert_enriched_chunks(self, vectors: List[dict]) -> int:
        return self._batched_upsert(vectors, V2_NAMESPACE_ENRICHED)

    def upsert_project_masters(self, vectors: List[dict]) -> int:
        return self._batched_upsert(vectors, V2_NAMESPACE_PROJECT_MASTER)

    def _batched_upsert(self, vectors: List[dict], namespace: str) -> int:
        if not vectors:
            return 0
        total = 0
        for start in range(0, len(vectors), UPSERT_BATCH):
            batch = vectors[start : start + UPSERT_BATCH]
            self.pc.index.upsert(vectors=batch, namespace=namespace)
            total += len(batch)
        logger.info("v2 pinecone upsert", extra={"namespace": namespace, "n": total})
        return total

    # ───── delete (by file_id prefix or project_id) ─────
    def delete_chunks_by_file_id(self, file_id: str) -> int:
        return self._delete_by_prefix(f"{file_id}-", V2_NAMESPACE_ENRICHED)

    def delete_master_by_project_id(self, project_id: str) -> int:
        try:
            self.pc.index.delete(ids=[project_id], namespace=V2_NAMESPACE_PROJECT_MASTER)
            return 1
        except Exception as e:
            logger.warning("delete master failed", extra={"project_id": project_id, "err": str(e)})
            return 0

    def _delete_by_prefix(self, prefix: str, namespace: str) -> int:
        deleted = 0
        try:
            for id_batch in self.pc.index.list(prefix=prefix, namespace=namespace):
                if not id_batch:
                    continue
                for i in range(0, len(id_batch), 1000):
                    chunk = id_batch[i : i + 1000]
                    self.pc.index.delete(ids=chunk, namespace=namespace)
                    deleted += len(chunk)
        except Exception as e:
            logger.warning("delete by prefix failed", extra={"prefix": prefix, "namespace": namespace, "err": str(e)})
        return deleted

    # ───── stats ─────
    def stats(self) -> dict:
        try:
            s = self.pc.index.describe_index_stats()
            namespaces = s.get("namespaces", {}) or {}
            return {
                "v2_enriched_vectors": int((namespaces.get(V2_NAMESPACE_ENRICHED) or {}).get("vector_count", 0)),
                "v2_project_master_vectors": int((namespaces.get(V2_NAMESPACE_PROJECT_MASTER) or {}).get("vector_count", 0)),
                "v1_knowledgebase_vectors": int((namespaces.get("knowledgebase") or {}).get("vector_count", 0)),
                "all_namespaces": list(namespaces.keys()),
            }
        except Exception as e:
            logger.warning("stats failed", extra={"err": str(e)})
            return {}
