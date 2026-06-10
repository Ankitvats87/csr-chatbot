"""Quick CLI to ask V2 (or V1) any question and see the retrieved chunks + LLM answer.

Usage:
    python -u scripts/ask_v2.py "What is the budget for the Nirogya project?"
    python -u scripts/ask_v2.py --v1 "What is the budget for the Nirogya project?"   # force V1
    python -u scripts/ask_v2.py --no-llm "Which projects are in healthcare?"         # retrieval only
    python -u scripts/ask_v2.py --top 5 "your question"                              # show top N chunks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.services.embedding_service import EmbeddingService
from app.services.prompt_service import PromptService
from app.services.response_service import ResponseService
from app.services.vector_service import VectorService
from app.services.vector_service_v2 import VectorServiceV2
from app.utils.env import get_settings
from app.utils.logger import setup_logging


def main() -> int:
    p = argparse.ArgumentParser(description="Ask a question against V1 or V2 retrieval.")
    p.add_argument("question", type=str, help="Question to ask")
    p.add_argument("--v1", action="store_true", help="Force V1 retrieval (overrides RAG_VERSION).")
    p.add_argument("--v2", action="store_true", help="Force V2 retrieval (overrides RAG_VERSION).")
    p.add_argument("--no-llm", action="store_true", help="Show retrieved chunks only, skip LLM answer.")
    p.add_argument("--top", type=int, default=10, help="Show top N retrieved chunks (default 10).")
    args = p.parse_args()

    s = get_settings()
    setup_logging("WARNING")

    pc = PineconeClient(s)
    pc.connect()

    # Decide V1 vs V2
    if args.v1:
        version = "v1"
    elif args.v2:
        version = "v2"
    else:
        version = s.rag_version

    if version == "v2":
        svc = VectorServiceV2(pc, s)
    else:
        svc = VectorService(pc, s)

    emb = EmbeddingService(s)

    print(f"\n=== Asking [{version.upper()}] ===")
    print(f"Q: {args.question}\n")

    embedding = emb.embed(args.question)
    chunks = svc.query(embedding, question=args.question) if version == "v2" else svc.query(embedding)

    print(f"Retrieved {len(chunks)} chunks (showing top {min(args.top, len(chunks))}):\n")
    for i, c in enumerate(chunks[: args.top], 1):
        doc = c.document_name or "unknown"
        page = f" p.{c.page}" if c.page else ""
        preview = c.text[:240].replace("\n", " ").strip()
        print(f"[{i}] ({c.score:.2f}) {doc}{page}")
        print(f"    {preview}...")
        print()

    if args.no_llm or not chunks:
        return 0

    # LLM answer
    prompts = PromptService()
    responder = ResponseService(s)
    messages = prompts.build_messages(
        question=args.question,
        chunks=chunks,
        memory=[],
        document_directory="",
    )
    print("=== LLM answer ===")
    llm = responder.generate(messages)
    print(f"\n{llm.text}\n")
    print(f"(provider={llm.provider}, model={llm.model})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
