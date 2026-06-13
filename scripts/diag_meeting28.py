#!/usr/bin/env python3
"""VPS-only diagnostic: what does Pinecone actually return for meeting-28 queries?

Run ON the VPS (this sandbox can't reach api.pinecone.io):

    docker exec csrbot-app python scripts/diag_meeting28.py

Prints: index stats, top-15 chunks per namespace, metadata fields, text snippets.
Copy-paste the full output and send it back — it's the ground truth we need.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.env import get_settings
from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService

QUERIES = [
    "date of 28th CSR committee meeting",
    "28th CSR meeting minutes",
    "when was the 28th CSR committee meeting held",
]

NAMESPACES = ["csr_v2_enriched", "knowledgebase", "csr_project_master"]

TOP_K = 15

SEP = "=" * 72


def main():
    settings = get_settings()
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Embedding model: {settings.openai_embedding_model}")
    print(SEP)

    # --- Connect Pinecone ---
    pc = PineconeClient(settings)
    pc.connect()
    if not pc.health_ok():
        print("FATAL: Pinecone not reachable. Aborting.")
        sys.exit(1)
    print("Pinecone connected OK")

    # --- Index stats ---
    stats = pc.index.describe_index_stats()
    print(f"\n{SEP}\nINDEX STATS\n{SEP}")
    print(f"Total vectors: {stats.get('total_vector_count', '?')}")
    ns_map = stats.get("namespaces", {})
    for ns_name, ns_info in sorted(ns_map.items()):
        print(f"  namespace '{ns_name}': {ns_info.get('vector_count', '?')} vectors")

    # --- Embedding service ---
    emb_svc = EmbeddingService(settings)

    # --- Query each combination ---
    for query in QUERIES:
        print(f"\n{SEP}\nQUERY: \"{query}\"\n{SEP}")
        vec = emb_svc.embed(query)

        for ns in NAMESPACES:
            if ns not in ns_map:
                print(f"\n  [namespace '{ns}' does not exist — skipping]")
                continue
            print(f"\n  --- namespace: {ns} (top {TOP_K}) ---")

            # Query with metadata
            try:
                results = pc.index.query(
                    vector=vec,
                    top_k=TOP_K,
                    namespace=ns,
                    include_metadata=True,
                )
            except Exception as e:
                print(f"    QUERY ERROR: {e}")
                continue

            matches = results.get("matches", [])
            if not matches:
                print("    (no matches)")
                continue

            for i, m in enumerate(matches):
                score = m.get("score", "?")
                mid = m.get("id", "?")
                meta = m.get("metadata", {})
                source = meta.get("source", "?")
                meeting_num = meta.get("meeting_number", "NOT SET")
                doc_type = meta.get("document_type", "NOT SET")
                text = meta.get("text", "")
                snippet = text[:200].replace("\n", " ") if text else "(no text in metadata)"

                print(f"\n    [{i+1}] score={score:.4f}  id={mid[:40]}...")
                print(f"        source       = {source}")
                print(f"        meeting_num  = {meeting_num}")
                print(f"        doc_type     = {doc_type}")
                # Print ALL metadata keys so we see what's available
                other_keys = [k for k in sorted(meta.keys()) if k not in ("text", "source", "meeting_number", "document_type")]
                if other_keys:
                    for k in other_keys:
                        v = meta[k]
                        if isinstance(v, str) and len(v) > 100:
                            v = v[:100] + "..."
                        print(f"        {k:12s} = {v}")
                print(f"        text snippet = {snippet}")

    # --- Also check: does meeting_number=28 exist at all? ---
    print(f"\n{SEP}\nMETADATA FILTER: meeting_number == 28\n{SEP}")
    vec28 = emb_svc.embed("CSR committee meeting 28")
    for ns in NAMESPACES:
        if ns not in ns_map:
            continue
        print(f"\n  --- namespace: {ns} ---")
        try:
            results = pc.index.query(
                vector=vec28,
                top_k=5,
                namespace=ns,
                include_metadata=True,
                filter={"meeting_number": {"$eq": 28}},
            )
        except Exception as e:
            print(f"    FILTER ERROR: {e}")
            continue
        matches = results.get("matches", [])
        if not matches:
            print("    (no vectors with meeting_number=28)")
        for i, m in enumerate(matches):
            meta = m.get("metadata", {})
            print(f"    [{i+1}] score={m.get('score','?'):.4f}  source={meta.get('source','?')}  text={meta.get('text','')[:120]}...")

    # --- Check meeting_number distribution ---
    print(f"\n{SEP}\nMETADATA FILTER: meeting_number for each 20..30\n{SEP}")
    dummy_vec = vec28  # reuse
    for num in range(20, 31):
        for ns in ["csr_v2_enriched"]:
            if ns not in ns_map:
                continue
            try:
                results = pc.index.query(
                    vector=dummy_vec,
                    top_k=1,
                    namespace=ns,
                    include_metadata=True,
                    filter={"meeting_number": {"$eq": num}},
                )
                matches = results.get("matches", [])
                count_label = f"found (score={matches[0].get('score','?'):.4f})" if matches else "NOT FOUND"
                src = matches[0].get("metadata", {}).get("source", "") if matches else ""
                print(f"  meeting_number={num:2d}: {count_label}  {src[:60]}")
            except Exception as e:
                print(f"  meeting_number={num:2d}: ERROR {e}")

    print(f"\n{SEP}\nDONE — copy everything above and send it back.\n{SEP}")


if __name__ == "__main__":
    main()
