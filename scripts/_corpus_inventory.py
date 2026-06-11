"""One-shot corpus inventory — what's actually been extracted from PTC docs.
Used during Phase 1 to ground the benchmark questions in real data.
"""
import json, glob, collections, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = []
for p in glob.glob(os.path.join(ROOT, "data", "processed_v2", "*.json")):
    with open(p, encoding="utf-8") as f:
        try:
            docs.append((os.path.basename(p), json.load(f)))
        except Exception as e:
            print("err", p, e)

print(f"documents: {len(docs)}\n")
types = collections.Counter(d.get("document_type") for _, d in docs)
print("doc types:", dict(types))

meetings = [(name, d.get("meeting", {}).get("meeting_number"), d.get("meeting", {}).get("financial_year"), d.get("document_type")) for name, d in docs]
print("\n-- per-document meeting / FY / doc_type --")
for m in meetings: print(" ", m)

print("\nFY values:", sorted({m[2] for m in meetings if m[2]}))
print("Meeting numbers:", sorted({m[1] for m in meetings if m[1]}))

projs, ngos = [], set()
for name, d in docs:
    for p in d.get("projects", []):
        pn = p.get("project_name")
        n = p.get("ngo", {}).get("ngo_name") if isinstance(p.get("ngo"), dict) else None
        if n: ngos.add(n)
        if pn: projs.append((d.get("meeting", {}).get("meeting_number"), pn, n, (p.get("financial") or {}).get("project_cost") or (p.get("financial") or {}).get("approved_cost")))

print(f"\nunique projects: {len(set(p[1] for p in projs))}; NGOs: {len(ngos)}")
print("\n-- all projects (meeting, name, NGO, cost) --")
for x in projs: print(" ", x)
print("\n-- NGOs --")
for n in sorted(ngos): print(" ", n)
