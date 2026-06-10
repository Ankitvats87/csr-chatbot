"""Build data/v2_chunk_store.json from the LIVE csr_v2_enriched namespace.

The chunk store is the local BM25 corpus for hybrid retrieval. It is kept in
sync automatically by the V2 ingestion pipeline, but this script (re)builds it
from Pinecone directly — run it ONCE after deploying hybrid retrieval so the
already-ingested corpus becomes lexically searchable without re-ingestion:

    cd telegram-rag-bot
    python -u scripts/build_chunk_store.py

Read-only against Pinecone. Safe to re-run any time.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.pinecone_client import PineconeClient
from app.ingestion_v2.pinecone_v2 import V2_NAMESPACE_ENRICHED
from app.utils.env import get_settings
from app.utils.logger import setup_logging

FETCH_BATCH = 100


def main() -> int:
    settings = get_settings()
    setup_logging(settings.log_level)

    pinecone = PineconeClient(settings)
    pinecone.connect()
    index = pinecone.index

    print(f"Listing vector ids in namespace '{V2_NAMESPACE_ENRICHED}'...")
    all_ids: list[str] = []
    for id_batch in index.list(namespace=V2_NAMESPACE_ENRICHED):
        all_ids.extend(id_batch)
    print(f"  found {len(all_ids)} vectors")
    if not all_ids:
        print("Nothing to do — namespace is empty. Run V2 ingestion first.")
        return 1

    chunks: list[dict] = []
    for start in range(0, len(all_ids), FETCH_BATCH):
        batch_ids = all_ids[start : start + FETCH_BATCH]
        resp = index.fetch(ids=batch_ids, namespace=V2_NAMESPACE_ENRICHED)
        vectors = resp.vectors if hasattr(resp, "vectors") else resp.get("vectors", {})
        for vid, v in vectors.items():
            md = dict(v.metadata if hasattr(v, "metadata") else v.get("metadata", {}) or {})
            text = md.pop("text", "")
            md.pop("embedded_text_preview", None)
            chunks.append({"id": vid, "text": text, "metadata": md})
        print(f"  fetched {min(start + FETCH_BATCH, len(all_ids))}/{len(all_ids)}")

    p = Path(settings.chunk_store_path)
    store_path = p if p.is_absolute() else Path(__file__).resolve().parent.parent / p
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        json.dumps(
            {"built_at": datetime.utcnow().isoformat() + "Z", "chunks": chunks},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {len(chunks)} chunks to {store_path}")
    print("Hybrid BM25 retrieval is now ready — restart the app (or it loads on next boot).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
