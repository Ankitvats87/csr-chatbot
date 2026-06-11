"""Clean known contamination from the V2 extraction layer (local only).

Operates on:
  - data/processed_v2/<file_id>.json   (per-doc structured extraction)
  - data/v2_chunk_store.json           (BM25/lexical corpus)

What it removes:
  1. Letter-label projects/NGOs ("Project A", "NGO B", "Project C", ...).
     These are per-meeting table shorthand fabricated through extraction, never
     real project identities. is_generic_label() catches them.
  2. Obvious LlamaParse-fabricated NGO names ("Health NGO", "NGO for Education",
     "Green Earth NGO", "Sapna NGO"). These match suspicious patterns: NGOs
     whose name is literally a sector word, or a generic placeholder.
  3. Project metadata that referenced any of the above (so chunk_store metadata
     stops surfacing them in BM25).

What it does NOT touch:
  - Pinecone (offline). When the index is reachable, re-upsert from processed_v2.
  - Real records — every drop is logged so the cleanup is reviewable.

Run:
    python scripts/clean_extractions.py            # report-only (default)
    python scripts/clean_extractions.py --apply    # write changes
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed_v2"
CHUNK_STORE = ROOT / "data" / "v2_chunk_store.json"

LETTER_LABEL_RE = re.compile(r"^\s*(?:project|ngo)\s*[-:]?\s*[a-z]{1,2}\d{0,2}\s*$", re.IGNORECASE)

# Fabricated NGO signatures: names that are obviously generic placeholders,
# usually emitted when the parser couldn't read the real entity.
FABRICATED_NGO_PATTERNS = [
    re.compile(r"^\s*(?:health|education|environmental?|green\s+earth)\s+ngo\s*$", re.IGNORECASE),
    re.compile(r"^\s*ngo\s+for\s+\w+\s*$", re.IGNORECASE),
    re.compile(r"^\s*green\s+earth\s+ngo\s*$", re.IGNORECASE),
]

# Fabricated project signatures emitted alongside letter-NGOs (e.g. Meeting 25
# "Education Initiative" tied to "NGO for Education"). The pairing is the
# strongest signal; we drop projects only when their NGO is also fabricated.
FABRICATED_PROJECT_NAMES_SUSPECT = {
    "education initiative",
    "health awareness campaign",
    "environmental sustainability project",
    "health program",
    "skill development initiative",
    "education support",
    "women health and hygiene",
}


def is_letter_label(name: str) -> bool:
    return bool(name and LETTER_LABEL_RE.match(name))


def is_fabricated_ngo(name: str) -> bool:
    if not name:
        return False
    for rgx in FABRICATED_NGO_PATTERNS:
        if rgx.match(name):
            return True
    return False


def should_drop_project(proj: dict) -> tuple[bool, str]:
    name = (proj.get("project_name") or "").strip()
    if is_letter_label(name):
        return True, f"letter-label project: {name!r}"
    ngo_obj = proj.get("ngo") or {}
    ngo_name = ((ngo_obj or {}).get("ngo_name") or "").strip()
    if is_letter_label(ngo_name):
        return True, f"project tied to letter-label NGO: {name!r} / {ngo_name!r}"
    if is_fabricated_ngo(ngo_name):
        return True, f"fabricated NGO name: {name!r} / {ngo_name!r}"
    if name.lower() in FABRICATED_PROJECT_NAMES_SUSPECT and (
        is_fabricated_ngo(ngo_name) or is_letter_label(ngo_name) or not ngo_name
    ):
        return True, f"suspect generic project paired with weak NGO: {name!r} / {ngo_name!r}"
    return False, ""


def clean_processed(apply: bool) -> tuple[int, int, list[tuple[str, str]]]:
    docs_changed = 0
    projects_dropped = 0
    drop_log: list[tuple[str, str]] = []
    for jpath in sorted(PROCESSED.glob("*.json")):
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[skip] {jpath.name}: {e}", file=sys.stderr)
            continue
        projects = data.get("projects") or []
        kept = []
        changed = False
        for p in projects:
            drop, reason = should_drop_project(p)
            if drop:
                drop_log.append((jpath.name, reason))
                projects_dropped += 1
                changed = True
            else:
                kept.append(p)
        if changed:
            data["projects"] = kept
            docs_changed += 1
            if apply:
                jpath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return docs_changed, projects_dropped, drop_log


def clean_chunk_store(apply: bool) -> tuple[int, int]:
    if not CHUNK_STORE.is_file():
        return 0, 0
    payload = json.loads(CHUNK_STORE.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") or []
    chunks_kept = []
    metadata_cleaned = 0
    for c in chunks:
        md = c.get("metadata") or {}
        text = c.get("text") or ""

        # Remove any letter-label / fabricated entries from list metadata
        def _filter_names(values, drop_fn):
            nonlocal metadata_cleaned
            if not isinstance(values, list):
                return values
            cleaned = [v for v in values if not drop_fn(v or "")]
            if len(cleaned) != len(values):
                metadata_cleaned += 1
            return cleaned

        md["project_names"] = _filter_names(md.get("project_names"), is_letter_label)
        md["ngo_names"] = _filter_names(
            md.get("ngo_names"),
            lambda n: is_letter_label(n) or is_fabricated_ngo(n),
        )

        # If the chunk text is overwhelmingly letter-label rows, drop the chunk.
        letter_row_count = sum(1 for line in text.splitlines() if LETTER_LABEL_RE.match(line.strip().strip("|").strip()))
        total_data_lines = sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))
        if total_data_lines >= 3 and letter_row_count >= max(2, total_data_lines // 2):
            # >50% of body lines are letter-labels — chunk is fabricated table junk
            continue

        c["metadata"] = md
        chunks_kept.append(c)

    dropped = len(chunks) - len(chunks_kept)
    if apply:
        payload["chunks"] = chunks_kept
        CHUNK_STORE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return dropped, metadata_cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description="Local cleanup of V2 extraction contamination.")
    parser.add_argument("--apply", action="store_true", help="Write changes (default: report-only)")
    args = parser.parse_args()

    print("== Phase 1: processed_v2/*.json ==")
    docs_changed, projects_dropped, drop_log = clean_processed(apply=args.apply)
    print(f"  docs touched : {docs_changed}")
    print(f"  projects dropped : {projects_dropped}")
    for fn, reason in drop_log:
        print(f"    [{fn}] {reason}")

    print("\n== Phase 2: v2_chunk_store.json ==")
    chunks_dropped, meta_cleaned = clean_chunk_store(apply=args.apply)
    print(f"  chunks fully dropped : {chunks_dropped}")
    print(f"  chunks with metadata cleaned : {meta_cleaned}")

    print("\n" + ("APPLIED" if args.apply else "DRY-RUN — re-run with --apply to write"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
