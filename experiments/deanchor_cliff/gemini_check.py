"""Cross-family confirmation of the DE-ANCHORED cliff: does a third model family
(gemini-3-pro) also collapse on the cleanest generator (true_do de-anchored from
the visible-cell mean)? The uniform cliff's strength was 4 bases x 3 families; the
de-anchored (main-text) version so far only has gpt-5.1 + deepseek. This adds the
gemini family on the de-anchored SCMs.

Run: source ~/.theone_keys.env && python experiments/deanchor_cliff/gemini_check.py
"""
from __future__ import annotations
import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_d = importlib.util.spec_from_file_location("deanchor", HERE / "run.py")
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)  # provides k_graph_skewed (=de-anchor)

KS = [4, 5, 6]
N = 15
TOL = 0.005
MODEL = "gemini-2.5-pro"


def ask_gemini(text, maxtok=24000):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={os.environ['GEMINI_API_KEY']}",
            data=json.dumps({"systemInstruction": {"parts": [{"text": R.SYS}]},
                             "contents": [{"parts": [{"text": text + R.PROTO}]}],
                             "generationConfig": {"temperature": 0, "maxOutputTokens": maxtok}}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        c = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "fail": str(e)[:80]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1),
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "gemini_rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["k"], r["i"]))
    jf = jpath.open("a")
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            g = D.k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)   # de-anchor generator
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
            gem = ask_gemini(R.render(g, k))
            row = {"k": k, "i": i, "truth": truth, "gemini": gem}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.3f} gemini={gem.get('pred')} ({gem.get('latency')}s)", flush=True)
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== gemini-3-pro on DE-ANCHORED cliff ===")
    for k in KS:
        kr = [r for r in rows if r["k"] == k]; gp = [r for r in kr if r["gemini"]["pred"] is not None]
        acc = np.mean([1 if abs(r["gemini"]["pred"] - r["truth"]) <= TOL else 0 for r in gp]) if gp else None
        mae = np.mean([abs(r["gemini"]["pred"] - r["truth"]) for r in gp]) if gp else None
        fail = sum(1 for r in kr if r["gemini"]["pred"] is None)
        print(f"k={k}: acc={round(acc,3) if acc is not None else None} mae={round(mae,4) if mae is not None else None} protocol_fail={fail}/{len(kr)}")


if __name__ == "__main__":
    main()
