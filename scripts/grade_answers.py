"""End-to-end ANSWER accuracy grading (LLM-as-judge).

evaluate_v2.py measures retrieval (did the right chunks come back?).
This script measures what the user actually experiences: it runs questions
through the FULL intelligence-layer pipeline (planner → hybrid retrieval →
rerank → dual-pass generation) and has an LLM judge grade each answer for
groundedness and relevance against the retrieved evidence.

Usage:
    cd telegram-rag-bot
    python -u scripts/grade_answers.py                 # 25-question sample
    python -u scripts/grade_answers.py --sample 50
    python -u scripts/grade_answers.py --out grades.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.pinecone_client import PineconeClient
from app.db.sqlite_client import SQLiteClient
from app.services.document_directory_service import DocumentDirectoryService
from app.services.embedding_service import EmbeddingService
from app.services.intelligence_layer import IntelligenceLayerService
from app.services.response_service import ResponseService
from app.services.vector_service_v2 import VectorServiceV2
from app.utils.env import get_settings
from app.utils.logger import setup_logging

from evaluate_v2 import BENCHMARK  # reuse the 100-question benchmark

JUDGE_PROMPT = """You are grading a CSR document assistant's answer.

QUESTION:
{question}

RETRIEVED EVIDENCE (what the assistant was allowed to use):
{context}

ASSISTANT'S ANSWER:
{answer}

Grade the answer on these criteria:
1. GROUNDED: every factual claim (names, numbers, statuses, dates) is supported by the evidence. Saying "no record found" when evidence is absent counts as grounded.
2. RELEVANT: it actually addresses the question asked.
3. NO_HALLUCINATION: no invented projects, amounts, meetings, or NGOs.

Respond with JSON only:
{{"grounded": true/false, "relevant": true/false, "no_hallucination": true/false, "verdict": "PASS" or "FAIL", "reason": "one short sentence"}}
verdict is PASS only if all three criteria are true."""


def build_layer(settings):
    sqlite = SQLiteClient(settings.sqlite_path)
    sqlite.connect()
    pinecone = PineconeClient(settings)
    pinecone.connect()

    embedder = EmbeddingService(settings)
    responder = ResponseService(settings)
    directory = DocumentDirectoryService(sqlite)

    hybrid = None
    if settings.enable_hybrid_retrieval:
        from app.services.hybrid_search import HybridSearchService
        hybrid = HybridSearchService(settings)
        print(f"Hybrid retrieval: {'ACTIVE' if hybrid.available else 'chunk store MISSING (vector-only)'}")

    reranker = None
    if settings.enable_reranker:
        from app.services.reranker import LLMReranker
        reranker = LLMReranker(responder)

    vectors = VectorServiceV2(pinecone, settings, hybrid=hybrid)
    layer = IntelligenceLayerService(
        embedder=embedder,
        vectors=vectors,
        responder=responder,
        directory=directory,
        settings=settings,
        hybrid=hybrid,
        reranker=reranker,
    )
    return layer, responder


def judge(responder: ResponseService, question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context[:12000], answer=answer)
    try:
        resp = responder.generate([{"role": "system", "content": prompt}])
        m = re.search(r"\{.*\}", resp.text, re.DOTALL)
        return json.loads(m.group(0)) if m else {"verdict": "ERROR", "reason": "no JSON from judge"}
    except Exception as e:
        return {"verdict": "ERROR", "reason": str(e)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade end-to-end answer accuracy with an LLM judge.")
    parser.add_argument("--sample", type=int, default=25, help="Number of benchmark questions (default 25).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    random.seed(args.seed)
    questions = random.sample(BENCHMARK, min(args.sample, len(BENCHMARK)))

    layer, responder = build_layer(settings)

    results = []
    n_pass = n_fail = n_err = 0
    for i, (q, _kws) in enumerate(questions, 1):
        try:
            rag_result = asyncio.run(layer.answer(chat_id=0, question=q, history=[]))
            context = "\n".join(f"[{c.document_name}] {c.text[:600]}" for c in rag_result.chunks[:20])
            grade = judge(responder, q, context or "(no evidence retrieved)", rag_result.answer)
        except Exception as e:
            grade = {"verdict": "ERROR", "reason": str(e)}
            rag_result = None

        v = grade.get("verdict", "ERROR")
        if v == "PASS":
            n_pass += 1
        elif v == "FAIL":
            n_fail += 1
        else:
            n_err += 1
        print(f"[{i:2d}/{len(questions)}] {v:5s}  {q[:70]}")
        if v != "PASS":
            print(f"          reason: {grade.get('reason', '')}")
        results.append({
            "question": q,
            "answer": rag_result.answer if rag_result else None,
            "n_chunks": len(rag_result.chunks) if rag_result else 0,
            "grade": grade,
        })
        time.sleep(0.2)

    graded = n_pass + n_fail
    print(f"\n{'='*60}")
    print(f"  Answer Accuracy : {n_pass / graded:.1%}  ({n_pass}/{graded} graded)" if graded else "  No questions graded.")
    print(f"  Errors          : {n_err}")
    print(f"{'='*60}")

    if args.out:
        Path(args.out).write_text(json.dumps({
            "answer_accuracy": round(n_pass / graded, 4) if graded else None,
            "passed": n_pass, "failed": n_fail, "errors": n_err,
            "results": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
