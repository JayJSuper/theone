"""lit_radar.py — the lean kernel's second half: LITERATURE RADAR.

Free only. No Perplexity, no SerpAPI, no Elicit. Two free endpoints:
  * arXiv API           (http://export.arxiv.org/api/query)   — newest preprints
  * Semantic Scholar    (api.semanticscholar.org/graph/v1)    — abstracts + meta

Queries our exact research keywords, dedups against a persistent seen-set, and
appends only NEW hits to a dated markdown digest. Idempotent: run it daily (cron)
or by hand; it never double-reports a paper.

CLI:  python tools/lit_radar.py            # default keyword set
      python tools/lit_radar.py "exact phrase you care about"
"""
from __future__ import annotations
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = Path(__file__).parent.parent / "docs" / "lit_radar"
SEEN = OUT / "seen.json"
DIGEST = OUT / "digest.md"

# the axes that define our novelty claim. Each query is a LIST of phrases that
# ALL must appear (AND semantics) — keeps results on-topic instead of newest-by-
# date noise. Tune freely.
QUERIES = [
    ["causal inference", "language model"],
    ["causal reasoning", "large language model", "benchmark"],
    ["probabilistic inference", "marginalization", "language model"],
    ["interventional", "Bayesian network", "language model"],
    ["verifiable", "causal", "reasoning"],
]
UA = {"User-Agent": "theone-lit-radar/1.0 (research)"}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def from_arxiv(query: list[str], n: int = 8) -> list[dict]:
    expr = " AND ".join(f'all:"{p}"' for p in query)
    q = urllib.parse.quote(expr)
    url = (f"http://export.arxiv.org/api/query?search_query={q}"
           f"&sortBy=relevance&max_results={n}")
    try:
        root = ET.fromstring(_get(url))
    except Exception as e:  # noqa: BLE001
        return [{"_error": f"arxiv: {str(e)[:120]}"}]
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        aid = (e.findtext("a:id", "", ns) or "").rsplit("/", 1)[-1]
        out.append({
            "id": f"arxiv:{aid}", "source": "arXiv",
            "title": " ".join((e.findtext("a:title", "", ns)).split()),
            "year": (e.findtext("a:published", "", ns) or "")[:4],
            "url": e.findtext("a:id", "", ns),
            "abstract": " ".join((e.findtext("a:summary", "", ns)).split())[:400],
        })
    return out


def from_s2(query: list[str], n: int = 8) -> list[dict]:
    fields = "title,year,url,abstract,externalIds"
    url = ("https://api.semanticscholar.org/graph/v1/paper/search?"
           f"query={urllib.parse.quote(' '.join(query))}&limit={n}&fields={fields}")
    try:
        data = json.loads(_get(url))
    except Exception as e:  # noqa: BLE001 — S2 rate-limits anonymous calls
        return [{"_error": f"s2: {str(e)[:120]}"}]
    out = []
    for p in data.get("data", []):
        ext = p.get("externalIds") or {}
        pid = ext.get("ArXiv") and f"arxiv:{ext['ArXiv']}" or f"s2:{p.get('paperId')}"
        out.append({
            "id": pid, "source": "S2", "title": p.get("title", ""),
            "year": str(p.get("year") or ""), "url": p.get("url", ""),
            "abstract": (p.get("abstract") or "")[:400],
        })
    return out


def run(queries: list[str], today: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set(json.loads(SEEN.read_text())) if SEEN.exists() else set()
    fresh, errors = [], []
    for q in queries:
        for fn in (from_arxiv, from_s2):
            for hit in fn(q):
                if "_error" in hit:
                    errors.append(hit["_error"]); continue
                if hit["id"] in seen or hit["id"] in {h["id"] for h in fresh}:
                    continue
                hit["matched"] = " + ".join(q)
                fresh.append(hit)
            time.sleep(1)  # be polite to free endpoints
    if fresh:
        lines = [f"\n## {today}  ({len(fresh)} new)\n"]
        for h in fresh:
            lines.append(f"- **{h['title']}** ({h['year']}, {h['source']})  "
                         f"[{h['id']}]({h['url']})\n  - _matched:_ `{h['matched']}`\n"
                         f"  - {h['abstract']}\n")
            seen.add(h["id"])
        with DIGEST.open("a") as f:
            f.write("".join(lines))
        SEEN.write_text(json.dumps(sorted(seen)))
    return {"new": len(fresh), "total_seen": len(seen),
            "errors": sorted(set(errors)), "digest": str(DIGEST)}


if __name__ == "__main__":
    qs = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else QUERIES
    # date passed explicitly (no Date.now in this env's spirit); use file mtime-free stamp
    import datetime
    today = datetime.date.today().isoformat()
    r = run(qs, today)
    print(f"lit_radar: {r['new']} new / {r['total_seen']} seen "
          f"-> {r['digest']}")
    if r["errors"]:
        print("non-fatal endpoint errors:", "; ".join(r["errors"]))
