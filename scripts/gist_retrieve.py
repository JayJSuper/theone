"""Operator-side retrieval for the SSH-free cloud pipeline. Lists the gists owned by the GIST_TOKEN
and prints the most recent theone-result gist (the pod created it on completion). No pod access.

Usage:  GIST_TOKEN=<gist-scope-token> .venv/bin/python scripts/gist_retrieve.py [tag]
"""
import json, os, sys, urllib.request

TOKEN = os.environ.get("GIST_TOKEN", "")
if not TOKEN:
    raise SystemExit("set GIST_TOKEN (a GitHub token with gist scope)")
tag = sys.argv[1] if len(sys.argv) > 1 else None


def api(path):
    req = urllib.request.Request("https://api.github.com" + path, headers={
        "Authorization": "Bearer " + TOKEN, "Accept": "application/vnd.github+json",
        "User-Agent": "theone-operator"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


gists = api("/gists?per_page=30")
hits = [g for g in gists if (g.get("description") or "").startswith("theone-result")
        and (tag is None or tag in (g.get("description") or ""))]
if not hits:
    print("no theone-result gist yet (pod still running?).")
    raise SystemExit(0)
g = hits[0]                                              # most recent first
print(f"=== {g['description']} · {g['created_at']} · {g['html_url']} ===\n")
full = api("/gists/" + g["id"])
for fname, f in full["files"].items():
    print(f"---- {fname} ----")
    print(f.get("content", ""))
