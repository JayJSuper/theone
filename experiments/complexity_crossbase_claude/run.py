"""Claude cross-base at the cliff (Q-C21 attack-4): run claude on the SAME k=4
and k=5 problems (seed-reproduced from complexity_axis) to test whether the
combinatorial cliff is base-universal. claude-fable-5 is access-gated -> use
claude-opus-4-8 (the strongest available). Same 4096 budget, AM-007 scoring.
Run: source ~/.theone_keys.env && python .../run.py
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "ca", HERE.parent / "complexity_axis" / "run.py")
ca = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(ca)
from theone.causal.engine import InterventionEngine

MODEL = "claude-opus-4-8"
KS = [4, 5]
N_PER_K = 50
TOL = 0.005


def ask_claude(text):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": MODEL, "max_tokens": 4096,
                             "system": ca.SYS,
                             "messages": [{"role": "user",
                                           "content": text + ca.PROTO}]}).encode(),
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = "".join(b.get("text", "") for b in out.get("content", []))
        tok = out.get("usage", {}).get("output_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1),
                "tokens": 0, "fail": str(e)[:90]}
    m = None
    for m in re.finditer(r"ANSWER:\s*([0-9]*\.?[0-9]+)", c, re.I):
        pass
    return {"pred": float(m.group(1)) if m else None,
            "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if m else "no ANSWER"}


def main():
    jpath = HERE / "rows.jsonl"
    done = {(json.loads(l)["k"], json.loads(l)["i"])
            for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jsonl = jpath.open("a")
    rows = []
    for k in KS:
        for i in range(N_PER_K):
            g = ca.k_graph(k, ca.BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            if (k, i) in done:
                continue
            cl = ask_claude(ca.render(g, k))
            row = {"k": k, "i": i, "truth": truth, "claude": cl}
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.4f} claude={cl['pred']}"
                  f"({cl['latency']}s)", flush=True)
    allrows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n===== CLAUDE CROSSBASE (cliff k4/k5) =====")
    for k in KS:
        kr = [r for r in allrows if r["k"] == k]
        if not kr:
            continue
        acc = sum(1 for r in kr if r["claude"]["pred"] is not None
                  and abs(r["claude"]["pred"] - r["truth"]) <= TOL) / len(kr)
        fail = sum(1 for r in kr if r["claude"]["pred"] is None)
        print(f"k={k} (2^{k}={2**k}): claude acc={acc:.2f}  protocol_fail={fail}  "
              f"[ref: gpt-5.1 {'0.66' if k==4 else '0.08'}, deepseek "
              f"{'0.06' if k==4 else '0.00'}, engine 1.00]")


if __name__ == "__main__":
    main()
