from typing import List, Tuple

from app.ingestion.metadata_builder import build_chunk_metadata, vector_id
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.utils.logger import get_logger

logger = get_logger(__name__)

BATCH_SIZE = 64  # OpenAI embedding batch + Pinecone upsert batch


def embed_and_upsert(
    *,
    embedder: EmbeddingService,
    vectors: VectorService,
    file_id: str,
    document_name: str,
    upload_date: str,
    chunks: List[Tuple[str, int]],
) -> int:
    """Returns the number of chunks upserted."""
    # Idempotent re-index: wipe any prior vectors for this file.
    # delete_by_file_id paginates IDs by prefix and deletes synchronously,
    # so no sleep is needed.
    try:
        vectors.delete_by_file_id(file_id)
    except Exception as e:
        logger.debug("delete_by_file_id noop or failed", extra={"file_id": file_id, "err": str(e)})


    total = 0
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [t for t, _ in batch]
        embeddings = embedder.embed_batch(texts)
        payload = []
        for i, ((text, page), embedding) in enumerate(zip(batch, embeddings)):
            chunk_index = start + i
            payload.append(
                {
                    "id": vector_id(file_id, page, chunk_index),
                    "values": embedding,
                    "metadata": build_chunk_metadata(
                        file_id=file_id,
                        document_name=document_name,
                        page=page,
                        chunk_index=chunk_index,
                        text=text,
                        upload_date=upload_date,
                    ),
                }
            )
        vectors.upsert(payload)
        total += len(payload)
    logger.info(
        "upserted chunks",
        extra={"file_id": file_id, "document_name": document_name, "n": total},
    )
    return total
