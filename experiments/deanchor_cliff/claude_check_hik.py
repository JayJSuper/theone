"""4th base on the de-anchored (cleanest) generator: claude. The uniform cliff had
4 bases x 3 families; the de-anchored main-result so far has gpt-5.1/deepseek/gemini.
This adds claude so the headline figure matches the uniform cliff's cross-family
strength on the artifact-free generator. Same budget (4096) as gpt/deepseek for parity.

Run: source ~/.theone_keys.env && python experiments/deanchor_cliff/claude_check.py
"""
from __future__ import annotations
import importlib.util, json, os, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_d = importlib.util.spec_from_file_location("deanchor", HERE / "run.py")
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)

KS = [7, 8]
N = 12
TOL = 0.005
MODEL = "claude-opus-4-8"


def ask_claude(text, maxtok=4096):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": MODEL, "max_tokens": maxtok, "system": R.SYS,
                             "messages": [{"role": "user", "content": text + R.PROTO}]}).encode(),
            headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                     "x-api-key": os.environ["ANTHROPIC_API_KEY"]}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = "".join(b.get("text", "") for b in out.get("content", []))
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "fail": str(e)[:80]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1),
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "claude_rows_hik.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                done.add((json.loads(l)["k"], json.loads(l)["i"]))
    jf = jpath.open("a")
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            g = D.k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
            cl = ask_claude(R.render(g, k))
            row = {"k": k, "i": i, "truth": truth, "claude": cl}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.3f} claude={cl.get('pred')} ({cl.get('latency')}s)", flush=True)
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== claude on DE-ANCHORED cliff ===")
    for k in KS:
        kr = [r for r in rows if r["k"] == k]; gp = [r for r in kr if r["claude"]["pred"] is not None]
        acc = np.mean([abs(r["claude"]["pred"] - r["truth"]) <= TOL for r in gp]) if gp else None
        fail = sum(1 for r in kr if r["claude"]["pred"] is None)
        print(f"k={k}: acc={round(acc,3) if acc is not None else None} protocol_fail={fail}/{len(kr)}")


if __name__ == "__main__":
    main()
