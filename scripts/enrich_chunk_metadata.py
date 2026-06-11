"""Add natural-language synonyms to chunk metadata so BM25 hits by INTENT.

BM25 tokenizes metadata text. Many query intents ("board approval", "fund
release", "resolution") never appear verbatim in the raw chunk text, even when
the chunk IS about that exact thing — the source document just uses different
phrasing. We close that gap by mapping structural metadata (document_type,
lifecycle_stage, meeting_number) to a small set of synthetic search phrases
and adding them to the chunk's BM25 corpus via a new `search_hints` field.

`HybridSearchService._searchable_text` already concatenates known metadata
fields into the BM25 corpus — extending it to read `search_hints` is a
one-line change made elsewhere.

Idempotent.

Run:
    python scripts/enrich_chunk_metadata.py            # report
    python scripts/enrich_chunk_metadata.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNK_STORE = ROOT / "data" / "v2_chunk_store.json"

# document_type → list of phrases users would type
DOC_TYPE_HINTS = {
    "Board Minutes": ["board minutes", "board of directors", "bod", "board approval", "approved by board"],
    "Board Agenda": ["board agenda", "bod agenda"],
    "Resolution by Circulation": [
        "resolution by circulation", "rbc", "circulation",
        "board approval", "ratified by board", "resolution number",
    ],
    "CSR Agenda": ["csr agenda", "csr committee agenda", "agenda items"],
    "CSR Minutes": [
        "csr minutes", "csr committee minutes", "minutes of meeting",
        "committee recommendation", "csr committee approved",
    ],
    "MOA": ["moa", "memorandum of agreement", "memorandum of association", "moa signed", "moa executed"],
    "Progress Report": ["progress report", "progress update", "status update"],
    "Completion Report": ["completion report", "project completed", "closure"],
    "Utilisation Certificate": ["utilisation certificate", "uc", "uc submission"],
    "Annual CSR Report": ["annual csr report", "csr obligation", "csr expenditure"],
}

# lifecycle_stage → phrases
STAGE_HINTS = {
    "Proposal": ["proposal", "proposed project", "for consideration"],
    "Committee_Recommendation": [
        "committee recommendation", "csr committee approved",
        "recommended to board", "committee approved",
    ],
    "Board_Approval": [
        "board approval", "approved by board", "ratified by board",
        "board resolution", "resolution by circulation",
    ],
    "MOA_Signed": ["moa signed", "moa executed", "memorandum signed"],
    "Fund_Release": [
        "fund release", "funds released", "disbursed", "disbursement",
        "tranche released", "installment", "payment released",
    ],
    "Amendment": ["amendment", "modification", "revised", "amended"],
    "Progress_Update": ["progress update", "status update", "implementation status"],
    "Completion": ["completion", "completed", "project closed", "closure"],
    "Unknown": [],
}


def hints_for_chunk(md: dict) -> list[str]:
    out: list[str] = []
    dt = md.get("document_type")
    if dt in DOC_TYPE_HINTS:
        out.extend(DOC_TYPE_HINTS[dt])
    stage = md.get("lifecycle_stage")
    if stage in STAGE_HINTS:
        out.extend(STAGE_HINTS[stage])
    mnum = md.get("meeting_number")
    if mnum not in (None, ""):
        try:
            n = int(float(mnum))
            out.extend([f"meeting {n}", f"{n}th meeting", f"csr meeting {n}", f"{n}th csr meeting"])
        except (TypeError, ValueError):
            pass
    fy = md.get("financial_year")
    if fy:
        out.extend([f"financial year {fy}", f"fy {fy}", str(fy)])
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if not CHUNK_STORE.is_file():
        print("no chunk store", file=sys.stderr)
        return 1
    payload = json.loads(CHUNK_STORE.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") or []

    enriched = 0
    by_doctype = Counter()
    by_stage = Counter()
    for c in chunks:
        md = c.get("metadata") or {}
        hints = hints_for_chunk(md)
        if hints:
            md["search_hints"] = hints
            c["metadata"] = md
            enriched += 1
            by_doctype[md.get("document_type")] += 1
            by_stage[md.get("lifecycle_stage")] += 1

    print(f"chunks enriched: {enriched} / {len(chunks)}")
    print("  by document_type:")
    for k, v in by_doctype.most_common():
        print(f"    {k}: {v}")
    print("  by lifecycle_stage:")
    for k, v in by_stage.most_common():
        print(f"    {k}: {v}")

    if args.apply:
        payload["chunks"] = chunks
        CHUNK_STORE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print("APPLIED")
    else:
        print("DRY-RUN — re-run with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
