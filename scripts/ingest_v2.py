"""CLI for the V2 enriched ingestion pipeline.

Usage:
    python scripts/ingest_v2.py --all
    python scripts/ingest_v2.py --file "26th CSR Agenda dt 04.03.2025 _ Meeting dated 10.03.2025.pdf"
    python scripts/ingest_v2.py --dry-run --file <name>     # parse + extract only, no Pinecone writes
    python scripts/ingest_v2.py --stats                     # report namespace sizes (no changes)

This script does NOT touch V1 namespace (`knowledgebase`). It writes only to
csr_v2_enriched and csr_project_master.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.ingestion_v2.pinecone_v2 import V2Pinecone
from app.ingestion_v2.pipeline import V2Pipeline
from app.utils.env import get_settings
from app.utils.logger import setup_logging, get_logger


def _print_stats(v2: V2Pinecone) -> None:
    stats = v2.stats()
    print("\n=== Pinecone namespace stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Process every file in data/raw_docs/")
    g.add_argument("--file", action="append", help="Document name (per SQLite) or file_id. May repeat.")
    g.add_argument("--stats", action="store_true", help="Print namespace stats and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run parse + extract only; skip Pinecone writes (still writes JSON/MD to data/processed_v2).",
    )
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("ingest_v2.cli")

    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    pinecone = PineconeClient(settings)
    pinecone.connect()
    v2 = V2Pinecone(pinecone)

    if args.stats:
        _print_stats(v2)
        return 0

    pipeline = V2Pipeline(settings=settings, pinecone=pinecone, sqlite=sqlite)
    if args.dry_run:
        # Disable Pinecone writes by replacing the upsert methods with no-ops.
        pipeline.v2.upsert_enriched_chunks = lambda v: 0  # type: ignore
        pipeline.v2.upsert_project_masters = lambda v: 0  # type: ignore
        logger.info("DRY RUN — no Pinecone writes will occur")

    only = None
    if args.file:
        only = list(args.file)

    counts = pipeline.run_on_all_raw_docs(only_filenames=only)

    print("\n=== V2 ingestion report ===")
    print(f"  files seen:                {counts.files_seen}")
    print(f"  files indexed OK:          {counts.files_indexed}")
    print(f"  files failed:              {counts.files_failed}")
    print(f"  enriched chunks upserted:  {counts.enriched_chunks_upserted}")
    print(f"  project masters built:     {counts.project_master_count}")
    print(f"  project masters upserted:  {counts.project_masters_upserted}")
    if counts.errors:
        print("\n  errors:")
        for name, err in counts.errors:
            print(f"    - {name}: {err[:160]}")
    print()
    _print_stats(v2)
    return 0 if counts.files_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
