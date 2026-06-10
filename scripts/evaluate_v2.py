"""V2 Evaluation Framework — CSR Knowledge Graph RAG

Compares retrieval quality between V1 (knowledgebase) and V2 (csr_v2_enriched +
csr_project_master) using 100 representative CSR benchmark questions.

Metrics computed per question:
  - hit        : at least one expected keyword found in retrieved text (binary)
  - recall     : fraction of expected keywords found
  - n_chunks   : number of chunks returned above threshold
  - top_score  : highest similarity score returned

Overall metrics (averaged across all questions):
  - Hit Rate   : fraction of questions with at least 1 keyword hit
  - Avg Recall : mean keyword recall across all questions
  - Avg Chunks : mean chunks retrieved

Usage:
    cd telegram-rag-bot
    python -u scripts/evaluate_v2.py                    # full 100-question run
    python -u scripts/evaluate_v2.py --sample 20        # random 20-question sample
    python -u scripts/evaluate_v2.py --v2-only          # skip V1 (saves API cost)
    python -u scripts/evaluate_v2.py --out report.json  # write JSON report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.pinecone_client import PineconeClient
from app.services.embedding_service import EmbeddingService
from app.services.vector_service import VectorService
from app.services.vector_service_v2 import VectorServiceV2
from app.utils.env import get_settings
from app.utils.logger import setup_logging, get_logger

# ── 100 benchmark questions ──────────────────────────────────────────────────
# Each entry: (question, [expected_keywords])
# Keywords are case-insensitive substrings that SHOULD appear in retrieved text.
# They are chosen to be distinctive enough to confirm the right chunk was fetched.

BENCHMARK: List[tuple[str, list[str]]] = [
    # ── Meeting identification ───────────────────────────────────────
    ("What was discussed in the 26th CSR committee meeting?", ["26th", "26"]),
    ("List agenda items from the 25th CSR meeting.", ["25th", "25"]),
    ("What resolutions were passed in the 27th CSR meeting?", ["27th", "27"]),
    ("Who chaired the 26th CSR committee meeting?", ["26"]),
    ("What was the date of the 25th CSR meeting?", ["25"]),
    ("Which projects were approved in the 24th CSR meeting?", ["24"]),
    ("What were the minutes of the 26th CSR committee meeting?", ["26"]),
    ("How many agenda items were there in the 25th CSR meeting?", ["25"]),
    ("What was the financial year for the 26th CSR meeting?", ["26"]),
    ("Which NGOs presented proposals in the 26th meeting?", ["26"]),

    # ── Project identification ───────────────────────────────────────
    ("What is the Healthcare Kiosk Initiative?", ["healthcare kiosk", "kiosk"]),
    ("Provide details about the Nirogya Life Line Foundation project.", ["nirogya", "life line"]),
    ("What projects were approved for healthcare sector?", ["healthcare", "health"]),
    ("Which projects are related to education CSR activities?", ["education", "school"]),
    ("List all projects implemented in Gautam Buddha Nagar.", ["gautam buddha nagar"]),
    ("What CSR projects are being implemented in Uttar Pradesh?", ["uttar pradesh", "UP"]),
    ("Which projects target women beneficiaries?", ["women"]),
    ("List all completed CSR projects.", ["completed", "completion"]),
    ("What are the ongoing CSR projects?", ["in progress", "ongoing"]),
    ("Which projects have received board approval?", ["board approval", "approved by board"]),

    # ── Financial queries ────────────────────────────────────────────
    ("What is the approved budget for the Nirogya Life Line Foundation project?", ["nirogya", "11534300", "11,53"]),
    ("What is the total CSR expenditure approved?", ["approved", "cost", "amount"]),
    ("Which project has the highest approved cost?", ["approved cost", "cost"]),
    ("How much has been disbursed to the healthcare kiosk project?", ["kiosk", "disburs"]),
    ("What is the balance amount for Nirogya Life Line Foundation?", ["nirogya", "balance"]),
    ("What was the approved budget in FY 2024-25?", ["2024-25", "2024"]),
    ("List all projects with approved costs above 1 crore.", ["crore", "100"]),
    ("What is the utilisation certificate status for CSR projects?", ["utilisation", "UC"]),
    ("How much total CSR budget was approved in the last financial year?", ["approved", "financial year"]),
    ("What is the Schedule VII budget allocation?", ["schedule vii", "schedule 7"]),

    # ── NGO queries ──────────────────────────────────────────────────
    ("What is Nirogya Life Line Foundation's registration details?", ["nirogya", "registration"]),
    ("Which NGOs are implementing CSR projects?", ["ngo", "implementing"]),
    ("What type of organisation is Nirogya Life Line Foundation?", ["nirogya", "section 8", "trust", "society"]),
    ("List all NGO partners for healthcare projects.", ["ngo", "healthcare", "health"]),
    ("What projects are being implemented by NGOs in UP?", ["uttar pradesh", "UP", "ngo"]),
    ("Which NGO is implementing the environment project?", ["environment", "ngo"]),
    ("How many NGOs are active in the CSR program?", ["ngo"]),
    ("What is the MOA status with implementing NGOs?", ["moa", "ngo"]),
    ("Which NGOs have signed MOAs?", ["moa", "signed", "executed"]),
    ("List NGOs implementing education projects.", ["education", "ngo"]),

    # ── Lifecycle / status queries ───────────────────────────────────
    ("Which projects have MOA signed?", ["moa", "signed", "executed"]),
    ("What projects are at the proposal stage?", ["proposal", "proposed"]),
    ("Which projects have funds released?", ["fund", "release", "disburs"]),
    ("What is the completion status of healthcare kiosk project?", ["kiosk", "complet", "status"]),
    ("List all projects that received committee recommendation.", ["committee", "recommend"]),
    ("Which projects have progress reports submitted?", ["progress report", "progress"]),
    ("What amendments were made to existing projects?", ["amendment", "modif"]),
    ("Which projects were ratified by the board?", ["board", "ratif", "approv"]),
    ("What is the implementation status of Nirogya project?", ["nirogya", "status", "implement"]),
    ("List projects approved by board in FY 2024-25.", ["board", "approv", "2024"]),

    # ── Geography queries ────────────────────────────────────────────
    ("Which CSR projects are in Noida?", ["noida", "gautam buddha"]),
    ("List projects implemented in Rajasthan.", ["rajasthan"]),
    ("What are the CSR activities in Delhi?", ["delhi"]),
    ("Which districts have active CSR projects?", ["district"]),
    ("List all states where CSR projects are operational.", ["state"]),
    ("What project is being implemented in Gautam Buddha Nagar district?", ["gautam buddha nagar"]),
    ("Are there any CSR projects in Maharashtra?", ["maharashtra"]),
    ("Which cities have benefited from CSR activities?", ["city", "district", "state"]),
    ("List projects in North India.", ["uttar pradesh", "delhi", "rajasthan", "haryana"]),
    ("What is the geographic spread of CSR activities?", ["state", "district", "geography"]),

    # ── Beneficiary queries ──────────────────────────────────────────
    ("How many beneficiaries are targeted under the healthcare kiosk project?", ["kiosk", "beneficiar"]),
    ("What is the total beneficiary count across all projects?", ["beneficiar", "count", "total"]),
    ("Which projects target children as beneficiaries?", ["children", "child", "student"]),
    ("List projects targeting farmers.", ["farmer", "agricultur"]),
    ("What type of beneficiaries does the education project target?", ["education", "beneficiar"]),
    ("How many patients will benefit from the healthcare project?", ["patient", "healthcare", "health"]),
    ("Which CSR projects target women empowerment?", ["women", "empowerment"]),
    ("What is the beneficiary type for Nirogya Life Line Foundation?", ["nirogya", "beneficiar"]),
    ("List projects with more than 1000 beneficiaries.", ["beneficiar"]),
    ("What communities are targeted by CSR activities?", ["community", "beneficiar"]),

    # ── CSR Classification queries ───────────────────────────────────
    ("Which Schedule VII clause covers healthcare projects?", ["schedule", "vii", "healthcare"]),
    ("What is the CSR sector for the Nirogya project?", ["nirogya", "sector", "healthcare"]),
    ("List all projects under Schedule VII clause (i).", ["schedule", "clause"]),
    ("Which projects fall under environment and sustainability?", ["environment", "sustainab"]),
    ("What education CSR activities are approved?", ["education", "approv"]),
    ("List projects under Schedule VII.", ["schedule vii", "schedule 7"]),
    ("Which projects relate to rural development?", ["rural", "development"]),
    ("What CSR activities address hunger and poverty?", ["hunger", "poverty", "food"]),
    ("Are there any projects for senior citizens?", ["senior", "elderly", "aged"]),
    ("List projects for differently-abled beneficiaries.", ["disable", "differently abled", "handicap"]),

    # ── Governance & committee queries ───────────────────────────────
    ("Who are the members of the CSR committee?", ["member", "committee", "CSR committee"]),
    ("What is the resolution number for the healthcare kiosk approval?", ["resolution", "kiosk"]),
    ("What board resolutions were passed for CSR?", ["board", "resolution"]),
    ("What was the agenda for the 26th CSR meeting?", ["agenda", "26"]),
    ("Which committee members attended the 26th meeting?", ["26", "attend", "present"]),
    ("What is the resolution by circulation number?", ["resolution by circulation", "circulation"]),
    ("Who approved the Nirogya Life Line Foundation project?", ["nirogya", "approv", "board", "committee"]),
    ("What action items came out of the 25th CSR meeting?", ["25", "action"]),
    ("How many projects were considered in the 26th CSR meeting?", ["26", "project"]),
    ("What is the governance structure of the CSR committee?", ["governance", "committee", "board"]),

    # ── Cross-document & synthesis queries ──────────────────────────
    ("Give a full timeline of the healthcare kiosk project.", ["kiosk", "healthcare"]),
    ("Summarize all projects approved across all CSR meetings.", ["approv", "meeting"]),
    ("What projects are mentioned in both agenda and minutes?", ["agenda", "minutes"]),
    ("Track the progress of Nirogya Life Line Foundation from proposal to completion.", ["nirogya"]),
    ("Which projects appear in multiple CSR meetings?", ["meeting"]),
    ("List all projects with both MOA signed and funds released.", ["moa", "fund"]),
    ("What is the current status of all healthcare projects?", ["healthcare", "status"]),
    ("Compare the budgets of all approved CSR projects.", ["budget", "cost", "approv"]),
    ("Which NGO has the most CSR projects?", ["ngo"]),
    ("What is the aggregate CSR spend for FY 2024-25?", ["2024", "spend", "expenditure", "approved"]),
]

assert len(BENCHMARK) == 100, f"Expected 100 questions, got {len(BENCHMARK)}"


# ── Result dataclasses ───────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    question: str
    expected_keywords: List[str]
    hit: bool = False
    recall: float = 0.0
    n_chunks: int = 0
    top_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class NamespaceReport:
    namespace: str
    questions_run: int = 0
    hit_rate: float = 0.0
    avg_recall: float = 0.0
    avg_chunks: float = 0.0
    avg_top_score: float = 0.0
    results: List[QuestionResult] = field(default_factory=list)


# ── Core evaluation logic ────────────────────────────────────────────────────

def _evaluate_one(
    question: str,
    keywords: List[str],
    embedder: EmbeddingService,
    vector_svc,
    is_v2: bool = False,
) -> QuestionResult:
    result = QuestionResult(question=question, expected_keywords=keywords)
    try:
        embedding = embedder.embed(question)
        if is_v2:
            chunks = vector_svc.query(embedding, question=question)
        else:
            chunks = vector_svc.query(embedding)

        result.n_chunks = len(chunks)
        if chunks:
            result.top_score = max(c.score for c in chunks)

        combined_text = " ".join(c.text for c in chunks).lower()
        matched = [kw for kw in keywords if kw.lower() in combined_text]
        result.matched_keywords = matched
        result.recall = len(matched) / len(keywords) if keywords else 0.0
        result.hit = len(matched) > 0

    except Exception as e:
        result.error = str(e)
        logger.warning("eval question failed", extra={"question": question[:60], "err": str(e)})

    return result


def _aggregate(results: List[QuestionResult], namespace: str) -> NamespaceReport:
    valid = [r for r in results if r.error is None]
    n = len(valid)
    if n == 0:
        return NamespaceReport(namespace=namespace, questions_run=len(results))
    return NamespaceReport(
        namespace=namespace,
        questions_run=len(results),
        hit_rate=sum(r.hit for r in valid) / n,
        avg_recall=sum(r.recall for r in valid) / n,
        avg_chunks=sum(r.n_chunks for r in valid) / n,
        avg_top_score=sum(r.top_score for r in valid) / n,
        results=results,
    )


def _print_report(report: NamespaceReport) -> None:
    print(f"\n{'='*60}")
    print(f"  Namespace : {report.namespace}")
    print(f"  Questions : {report.questions_run}")
    print(f"  Hit Rate  : {report.hit_rate:.1%}  (>=1 keyword found)")
    print(f"  Avg Recall: {report.avg_recall:.1%}  (keywords found / total)")
    print(f"  Avg Chunks: {report.avg_chunks:.1f}  (per question)")
    print(f"  Avg Score : {report.avg_top_score:.3f}  (top similarity)")
    print(f"{'='*60}")
    # Show 10 worst misses (hit=False)
    misses = [r for r in report.results if not r.hit and not r.error]
    if misses:
        print(f"\n  Worst misses ({min(10, len(misses))} of {len(misses)}):")
        for r in misses[:10]:
            print(f"    --  {r.question[:80]}")
            print(f"        expected: {r.expected_keywords}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate V1 vs V2 retrieval on 100 CSR benchmark questions.")
    parser.add_argument("--sample", type=int, default=0, help="Run only N random questions (0 = all 100).")
    parser.add_argument("--v2-only", action="store_true", help="Skip V1 evaluation (saves cost).")
    parser.add_argument("--v1-only", action="store_true", help="Skip V2 evaluation.")
    parser.add_argument("--hybrid", action="store_true", help="Also evaluate V2 + hybrid BM25 fusion (requires data/v2_chunk_store.json).")
    parser.add_argument("--out", type=str, default="", help="Write JSON report to this file path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample.")
    args = parser.parse_args()

    settings = get_settings()
    setup_logging(settings.log_level)

    questions = list(BENCHMARK)
    if args.sample > 0:
        random.seed(args.seed)
        questions = random.sample(questions, min(args.sample, len(questions)))
        print(f"Running {len(questions)}-question sample (seed={args.seed}).")
    else:
        print(f"Running full {len(questions)}-question benchmark.")

    pinecone = PineconeClient(settings)
    pinecone.connect()
    embedder = EmbeddingService(settings)

    reports: List[NamespaceReport] = []

    # ── V1 evaluation ────────────────────────────────────────────────
    if not args.v2_only:
        print("\n[V1] Evaluating against 'knowledgebase' namespace...")
        v1_svc = VectorService(pinecone, settings)
        v1_results: List[QuestionResult] = []
        for i, (q, kws) in enumerate(questions, 1):
            r = _evaluate_one(q, kws, embedder, v1_svc, is_v2=False)
            v1_results.append(r)
            status = "OK" if r.hit else "--"
            print(f"  V1 [{i:3d}/{len(questions)}] {status}  {q[:65]}")
            time.sleep(0.1)  # gentle rate limit
        v1_report = _aggregate(v1_results, "knowledgebase (v1)")
        _print_report(v1_report)
        reports.append(v1_report)

    # ── V2 evaluation ────────────────────────────────────────────────
    if not args.v1_only:
        print("\n[V2] Evaluating against 'csr_v2_enriched + csr_project_master' namespaces...")
        v2_svc = VectorServiceV2(pinecone, settings)
        v2_results: List[QuestionResult] = []
        for i, (q, kws) in enumerate(questions, 1):
            r = _evaluate_one(q, kws, embedder, v2_svc, is_v2=True)
            v2_results.append(r)
            status = "OK" if r.hit else "--"
            print(f"  V2 [{i:3d}/{len(questions)}] {status}  {q[:65]}")
            time.sleep(0.1)
        v2_report = _aggregate(v2_results, "csr_v2_enriched + csr_project_master (v2)")
        _print_report(v2_report)
        reports.append(v2_report)

    # ── V2 + Hybrid evaluation ───────────────────────────────────────
    if args.hybrid:
        from app.services.hybrid_search import HybridSearchService

        hybrid_settings = settings.model_copy(update={"enable_hybrid_retrieval": True})
        hybrid = HybridSearchService(hybrid_settings)
        if not hybrid.available:
            print("\n[HYBRID] SKIPPED — chunk store missing. Run: python scripts/build_chunk_store.py")
        else:
            print("\n[HYBRID] Evaluating V2 + BM25 fusion...")
            h_svc = VectorServiceV2(pinecone, hybrid_settings, hybrid=hybrid)
            h_results: List[QuestionResult] = []
            for i, (q, kws) in enumerate(questions, 1):
                r = _evaluate_one(q, kws, embedder, h_svc, is_v2=True)
                h_results.append(r)
                status = "OK" if r.hit else "--"
                print(f"  HY [{i:3d}/{len(questions)}] {status}  {q[:65]}")
                time.sleep(0.1)
            h_report = _aggregate(h_results, "v2 + hybrid BM25 fusion")
            _print_report(h_report)
            reports.append(h_report)

    # ── Comparison summary ───────────────────────────────────────────
    if len(reports) >= 2:
        base = reports[0]
        for other in reports[1:]:
            hit_delta = other.hit_rate - base.hit_rate
            recall_delta = other.avg_recall - base.avg_recall
            print(f"\n{'='*60}")
            print(f"  {base.namespace} -> {other.namespace} delta")
            print(f"  Hit Rate  : {hit_delta:+.1%}")
            print(f"  Avg Recall: {recall_delta:+.1%}")
            print(f"{'='*60}")

    # ── JSON output ──────────────────────────────────────────────────
    if args.out:
        payload = {
            "benchmark_size": len(questions),
            "reports": [
                {
                    "namespace": r.namespace,
                    "questions_run": r.questions_run,
                    "hit_rate": round(r.hit_rate, 4),
                    "avg_recall": round(r.avg_recall, 4),
                    "avg_chunks": round(r.avg_chunks, 2),
                    "avg_top_score": round(r.avg_top_score, 4),
                    "per_question": [
                        {
                            "question": qr.question,
                            "hit": qr.hit,
                            "recall": round(qr.recall, 4),
                            "n_chunks": qr.n_chunks,
                            "top_score": round(qr.top_score, 4),
                            "matched": qr.matched_keywords,
                            "expected": qr.expected_keywords,
                            "error": qr.error,
                        }
                        for qr in r.results
                    ],
                }
                for r in reports
            ],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
