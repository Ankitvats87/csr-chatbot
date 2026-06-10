# Hybrid Retrieval Upgrade — NotebookLM-Parity Layer

Implemented 2026-06-10. This closes the accuracy gap identified in
`RETRIEVAL.md` / `Enhance Retrieval.md`: the intelligence layer (query planner,
entity routing, dual-pass generation) already existed, but retrieval was
dense-vector-only, so exact tokens — budget figures, resolution numbers, NGO
names, meeting ordinals — were frequently missed. NotebookLM never misses
those because it grounds on full source text.

## What was added

| Component | File | Purpose |
|---|---|---|
| BM25 + chunk store | `app/services/hybrid_search.py` | Pure-Python Okapi BM25 over a local JSON mirror of every `csr_v2_enriched` vector. No new pip dependencies. |
| RRF fusion | same file | Reciprocal Rank Fusion merges vector + BM25 rankings; chunks found by BOTH rise to the top. |
| Entity boost | same file | Chunks whose metadata matches planner-extracted entities (meeting number / project / NGO) get a score boost. |
| LLM reranker | `app/services/reranker.py` | One cheap listwise LLM call reorders the top candidates. Fails safe to original order. |
| Chunk store sync | `app/ingestion_v2/pipeline.py` | V2 ingestion now mirrors chunks into `data/v2_chunk_store.json` automatically. |
| Store rebuild | `scripts/build_chunk_store.py` | One-time build from live Pinecone (no re-ingestion needed). Read-only, safe to re-run. |
| Retrieval eval | `scripts/evaluate_v2.py --hybrid` | Adds a V2+hybrid pass to the 100-question retrieval benchmark. |
| Answer eval | `scripts/grade_answers.py` | End-to-end answer accuracy via LLM-as-judge through the full pipeline. |
| Ad-hoc CLI | `scripts/ask_intelligence.py` | Ask one question through the full pipeline from the terminal. |
| Bigger answers | `app/services/response_service.py` | `max_tokens` now configurable (`LLM_MAX_TOKENS`, default 1600 — was hardcoded 800, which truncated synthesis answers). |

## Query flow (when all flags on)

```
User question
→ Query Planner (intent + entities + rewrite)          [existing]
→ Entity resolution (fuzzy project/NGO match)          [existing]
→ Multi-hop Pinecone retrieval (filters per intent)    [existing]
→ BM25 lexical search over chunk store                 [NEW]
→ Reciprocal Rank Fusion                               [NEW]
→ Entity metadata boost                                [NEW]
→ LLM listwise rerank → top CONTEXT_MAX_CHUNKS         [NEW]
→ Dual-pass generation (draft → fact-check audit)      [existing]
```

## Feature flags (.env)

```env
ENABLE_INTELLIGENCE_LAYER=true   # planner + routing + dual-pass (existing)
ENABLE_HYBRID_RETRIEVAL=true     # BM25 + RRF fusion          (NEW)
ENABLE_RERANKER=true             # LLM listwise rerank        (NEW)
HYBRID_BM25_TOP_K=25
RERANK_CANDIDATES=30
CONTEXT_MAX_CHUNKS=24
CHUNK_STORE_PATH=data/v2_chunk_store.json
LLM_MAX_TOKENS=1600
```

Set `ENABLE_HYBRID_RETRIEVAL=false` (or both new flags) to restore the exact
previous behaviour. If the chunk store file is missing, hybrid silently
degrades to vector-only — nothing breaks.

## Local testing (Windows + Docker Desktop)

Docker Desktop cannot bind-mount Google Drive (G:) folders, so local testing
uses `Dockerfile.local` (bakes app + scripts + data into the image) and
`docker-compose.local.yml` (app + Telegram long-polling bridge):

```powershell
cd "G:\My Drive\AI\Projects\CSR Bot\CSR Chatbot\telegram-rag-bot"
docker compose -f docker-compose.local.yml up --build -d
docker logs -f csrbot-local-app      # wait for "Application startup complete"
```

Then message the bot from your phone. The app container auto-builds the chunk
store from live Pinecone at startup.

⚠️ The poller calls `deleteWebhook` on startup, which disconnects the
production VPS webhook (same bot token). After local testing:

```powershell
docker compose -f docker-compose.local.yml down
```

and on the VPS re-run `scripts/setup_webhook.sh` (or redeploy) to restore the
production webhook.

## Deploying to the VPS

1. Push/copy the repo to the VPS (includes new files above).
2. Add the new flags to the VPS `.env` (see block above).
3. `docker compose build && docker compose up -d`
4. One-time inside the app container:
   `docker exec csrbot-app python -u scripts/build_chunk_store.py`
   then `docker compose restart app`.
   (Future V2 ingestions keep the store in sync automatically.)
5. Re-arm the webhook if it was cleared: `bash scripts/setup_webhook.sh`

## Measuring accuracy

```bash
# Retrieval benchmark: v2 vs v2+hybrid (100 questions, keyword hit/recall)
python -u scripts/evaluate_v2.py --v2-only --hybrid --out eval_report.json

# End-to-end answer accuracy (LLM judge, full pipeline, 25-question sample)
python -u scripts/grade_answers.py --sample 25 --out grades.json
```

Targets from RETRIEVAL.md: retrieval >95%, answer accuracy >95%,
citation accuracy >98%, hallucination <2%.

## Verified locally (2026-06-10)

- 1462 chunks mirrored from `csr_v2_enriched`; hybrid ACTIVE at boot.
- "What projects were approved in the 26th CSR committee meeting?" →
  full project list with NGO names + ₹3.99 Cr budget + page-level citations.
- "Approved budget and status of Nirogya healthcare kiosk?" →
  ₹1,15,34,300, status, MOA date, location-change synthesis across TWO
  documents (BOD Minutes + 30th CSR Agenda) with citations.
