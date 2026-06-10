from typing import List

from app.db.pinecone_client import PineconeClient
from app.models.message_model import RetrievedChunk
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorService:
    def __init__(self, pinecone: PineconeClient, settings: Settings):
        self.pinecone = pinecone
        self.settings = settings

    def query(
        self,
        embedding: List[float],
        question: str = "",
        metadata_filter: dict | None = None,
        top_k: int | None = None,
    ) -> List[RetrievedChunk]:
        kwargs = {
            "vector": embedding,
            "top_k": top_k or self.settings.top_k,
            "namespace": self.settings.pinecone_namespace,
            "include_metadata": True,
        }
        if metadata_filter:
            kwargs["filter"] = metadata_filter
        try:
            resp = self.pinecone.index.query(**kwargs)
        except Exception as e:
            logger.exception("pinecone query failed", extra={"err": str(e), "filter": metadata_filter})
            return []

        matches = resp.get("matches") if isinstance(resp, dict) else getattr(resp, "matches", [])
        chunks: List[RetrievedChunk] = []
        for m in matches or []:
            score = m["score"] if isinstance(m, dict) else m.score
            if score is None or score < self.settings.similarity_threshold:
                continue
            md = m["metadata"] if isinstance(m, dict) else (m.metadata or {})
            md = md or {}
            chunks.append(
                RetrievedChunk(
                    text=md.get("text", ""),
                    score=float(score),
                    document_name=md.get("document_name"),
                    source=md.get("source"),
                    page=str(md.get("page")) if md.get("page") is not None else None,
                    chunk_id=md.get("chunk_id"),
                )
            )
        logger.info(
            "pinecone query",
            extra={
                "raw_matches": len(matches or []),
                "kept_after_threshold": len(chunks),
                "threshold": self.settings.similarity_threshold,
            },
        )
        return chunks

    def upsert(self, vectors: List[dict]) -> None:
        """vectors: [{id, values, metadata}, ...]. Used by Phase 2 ingestion."""
        if not vectors:
            return
        self.pinecone.index.upsert(vectors=vectors, namespace=self.settings.pinecone_namespace)

    def delete_by_file_id(self, file_id: str) -> int:
        """Delete every vector belonging to a Drive file.

        Pinecone *serverless* indexes do not support delete(filter=…) —
        they return:
            'Serverless indexes do not support deleting with metadata filtering'
        So we paginate the index for IDs starting with `<file_id>-` (vector_id
        format from metadata_builder.vector_id) and delete those in batches.

        Returns the count deleted (best-effort).
        """
        namespace = self.settings.pinecone_namespace
        deleted = 0
        try:
            # list() yields lists of IDs prefixed with `<file_id>-`.
            for id_batch in self.pinecone.index.list(
                prefix=f"{file_id}-", namespace=namespace
            ):
                if not id_batch:
                    continue
                # Pinecone caps delete batch size at ~1000.
                for i in range(0, len(id_batch), 1000):
                    chunk = id_batch[i : i + 1000]
                    self.pinecone.index.delete(ids=chunk, namespace=namespace)
                    deleted += len(chunk)
        except Exception as e:
            logger.exception(
                "delete_by_file_id failed",
                extra={"file_id": file_id, "namespace": namespace, "err": str(e)},
            )
        if deleted:
            logger.info(
                "deleted vectors by file_id",
                extra={"file_id": file_id, "n": deleted, "namespace": namespace},
            )
        return deleted
