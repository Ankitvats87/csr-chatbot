"""Augment chunk-store metadata with canonical NGO + project aliases.

CSR documents refer to the same NGO under multiple surface forms — e.g.
"Doctors for You (DFY)", "Doctors for You (DFY), Delhi", "DFY". BM25 ranks
those as different terms, so lexical retrieval misses obvious matches.

This script adds an `aliases` array to each chunk's metadata containing the
canonical form of every NGO/project name found in the chunk. The original
`ngo_names`/`project_names` lists are preserved — we ADD signal, not replace.

Idempotent. Re-runnable.

Run:
    python scripts/normalize_aliases.py            # report-only
    python scripts/normalize_aliases.py --apply
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed_v2"
CHUNK_STORE = ROOT / "data" / "v2_chunk_store.json"

_TRAILING_GEO = re.compile(r"[,\s]+(?:new\s+)?(?:delhi|noida|gurugram|mumbai|bengaluru|bangalore|kolkata|chennai|hyderabad|jaipur|lucknow|mandsaur|faridabad|haryana|uttar pradesh|rajasthan|bihar|karnataka|maharashtra)\s*\.?\s*$", re.IGNORECASE)
_HONORIFIC = re.compile(r"^\s*(?:shri|sri|smt|dr|mr|ms|the)\s+", re.IGNORECASE)
_PAREN = re.compile(r"\s*\([^)]*\)")
_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")


def canonical(name: str) -> str:
    """Aggressively normalize an NGO/project name for matching."""
    if not name:
        return ""
    s = name.strip()
    # Strip trailing geography
    s = _TRAILING_GEO.sub("", s).strip().strip(",.")
    # Strip honorifics
    s = _HONORIFIC.sub("", s)
    return s


def loose_key(name: str) -> str:
    """Aggressive lowercase tokens-sorted form for dedupe."""
    if not name:
        return ""
    s = canonical(name).lower()
    s = _PAREN.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    tokens = sorted(t for t in _WS.split(s) if t and len(t) > 1)
    return " ".join(tokens)


def build_alias_clusters(all_names: Iterable[str]) -> dict[str, Set[str]]:
    """Cluster surface variants of the same entity. Returns key -> {variants}."""
    clusters: dict[str, Set[str]] = defaultdict(set)
    for n in all_names:
        if not n:
            continue
        k = loose_key(n)
        if k:
            clusters[k].add(n.strip())
    # Also fold near-duplicates by token-set containment.
    keys = sorted(clusters.keys(), key=len, reverse=True)
    merged: dict[str, Set[str]] = {}
    used: set[str] = set()
    for k in keys:
        if k in used:
            continue
        merged_set = set(clusters[k])
        tokens_k = set(k.split())
        for other in keys:
            if other in used or other == k:
                continue
            tokens_o = set(other.split())
            # If one is a strict subset of the other (>=3 tokens), merge
            if tokens_o and tokens_k and (tokens_o.issubset(tokens_k) or tokens_k.issubset(tokens_o)):
                if min(len(tokens_o), len(tokens_k)) >= 2:
                    merged_set |= clusters[other]
                    used.add(other)
        used.add(k)
        merged[k] = merged_set
    return merged


def collect_all_names(field: str) -> set[str]:
    out: set[str] = set()
    for jpath in PROCESSED.glob("*.json"):
        try:
            d = json.loads(jpath.read_text(encoding="utf-8"))
        except Exception:
            continue
        for p in d.get("projects") or []:
            if field == "ngo":
                ngo = p.get("ngo") or {}
                if ngo.get("ngo_name"):
                    out.add(ngo["ngo_name"])
            elif field == "project":
                if p.get("project_name"):
                    out.add(p["project_name"])
                for a in p.get("project_aliases") or []:
                    if a:
                        out.add(a)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    ngo_clusters = build_alias_clusters(collect_all_names("ngo"))
    proj_clusters = build_alias_clusters(collect_all_names("project"))

    print(f"== NGO clusters ({len(ngo_clusters)}) ==")
    for k, vs in sorted(ngo_clusters.items()):
        if len(vs) > 1:
            print(f"  key={k!r}")
            for v in sorted(vs):
                print(f"    - {v}")

    print(f"\n== Project clusters ({len(proj_clusters)}) ==")
    multi = [(k, vs) for k, vs in proj_clusters.items() if len(vs) > 1]
    print(f"  multi-variant projects: {len(multi)}")
    for k, vs in sorted(multi)[:10]:
        print(f"  key={k[:60]!r}")
        for v in sorted(vs):
            print(f"    - {v[:90]}")

    # Reverse-lookup: any name → its full alias set
    def alias_set(name: str, clusters: dict[str, Set[str]]) -> Set[str]:
        k = loose_key(name)
        return clusters.get(k, {name}) if k else {name}

    if not CHUNK_STORE.is_file():
        print("no chunk store; nothing to update", file=sys.stderr)
        return 1
    payload = json.loads(CHUNK_STORE.read_text(encoding="utf-8"))
    chunks = payload.get("chunks") or []
    augmented = 0
    for c in chunks:
        md = c.get("metadata") or {}
        ngo_in = md.get("ngo_names") or []
        proj_in = md.get("project_names") or []
        if not (ngo_in or proj_in):
            continue
        all_ngo_aliases: Set[str] = set(ngo_in)
        for n in ngo_in:
            all_ngo_aliases |= alias_set(n, ngo_clusters)
        all_proj_aliases: Set[str] = set(proj_in)
        for p in proj_in:
            all_proj_aliases |= alias_set(p, proj_clusters)
        if all_ngo_aliases != set(ngo_in) or all_proj_aliases != set(proj_in):
            md["ngo_names"] = sorted(all_ngo_aliases)
            md["project_names"] = sorted(all_proj_aliases)
            augmented += 1
        c["metadata"] = md

    print(f"\nchunks augmented: {augmented}")
    if args.apply:
        payload["chunks"] = chunks
        CHUNK_STORE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        print("APPLIED")
    else:
        print("DRY-RUN — re-run with --apply")
    return 0


if __name__ == "__main__":
    sys.exit(main())
