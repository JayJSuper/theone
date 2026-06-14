"""Closes the §5.1 open question: does the scaffold help or HURT gpt-5.1 in ITS
struggling regime? The 12-node test showed no effect because gpt-5.1 wasn't
struggling there. Here we test minimal (SYS_A) vs scaffold (SYS_B) on the
DE-ANCHORED k=5 SCMs, where gpt-5.1 sits at ~0.04 accuracy (deep in the cliff).

Run: source ~/.theone_keys.env && python experiments/scaffold_crossbase/struggling.py
"""
from __future__ import annotations
import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_d = importlib.util.spec_from_file_location("deanchor", HERE.parent / "deanchor_cliff" / "run.py")
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)
_s = importlib.util.spec_from_file_location("scalerun", HERE.parent / "baseline_b_v1_scale" / "run.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)

K = 5
N = 25
TOL = 0.005


def ask_gpt(system, text, maxtok=4096):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text + R.PROTO}],
                "max_completion_tokens": maxtok}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]; tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:80]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "struggling_rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["i"])
    jf = jpath.open("a")
    for i in range(N):
        if i in done:
            continue
        g = D.k_graph_skewed(K, R.BASE_SEED + 1000 * K + i)   # de-anchored k=5 (struggling)
        truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
        text = R.render(g, K)
        a = ask_gpt(S.SYS_A, text); b = ask_gpt(S.SYS_B, text)
        row = {"i": i, "truth": truth, "A_minimal": a, "B_scaffold": b}
        jf.write(json.dumps(row) + "\n"); jf.flush()
        print(f"[{i:02d}] truth={truth:.3f} A={a.get('pred')} B={b.get('pred')}", flush=True)
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    def summ(arm):
        ok = [r for r in rows if r[arm]["pred"] is not None]
        acc = np.mean([abs(r[arm]["pred"] - r["truth"]) <= TOL for r in ok]) if ok else None
        mae = np.mean([abs(r[arm]["pred"] - r["truth"]) for r in ok]) if ok else None
        return (round(acc, 3) if acc is not None else None, round(mae, 4) if mae is not None else None,
                sum(1 for r in rows if r[arm]["pred"] is None))
    print(f"\n=== scaffold in gpt-5.1's STRUGGLING regime (de-anchored k=5, n={len(rows)}) ===")
    print(f"  A minimal  : acc/mae/fail = {summ('A_minimal')}")
    print(f"  B scaffold : acc/mae/fail = {summ('B_scaffold')}")
    (HERE / "struggling_results.json").write_text(json.dumps(
        {"n": len(rows), "A_minimal": summ("A_minimal"), "B_scaffold": summ("B_scaffold")}, indent=2, default=str))


if __name__ == "__main__":
    main()
