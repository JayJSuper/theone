"""Cross-base retest (Q-C15 attack-4 mandate): SAME 150 L-tier instances as the
formal run (seed-reproduced, NOT resampled), raw-A condition only, on gpt-5.1
and gemini-3.5-flash. Same 4096-token budget, same AM-007 scoring (protocol
failure = error, no retry), temperature 0 where the API allows.

PREREG (frozen at commit time, before any result lands):
- Instances: gen_dag(12 nodes), rng = default_rng(20260614+12), first 150 draws
  == the formal run's L tier exactly (gatekeeper: do not resample).
- Subjects: gpt-5.1 (OpenAI), gemini-3.5-flash (Google). Rationale: current
  flagship + same 'flash' tier as deepseek-v4-flash.
- Deviation notes: OpenAI GPT-5 series rejects non-default temperature -> omitted
  (affects determinism, not validity); both get 4096 completion tokens INCLUDING
  any thinking, exactly like deepseek in the formal run.
- Question: does the 12-node collapse generalize across bases? Verdict per base:
  accuracy within +-0.15 of deepseek's 0.613 => 'consistent'; > 0.763 => 'base-
  dependent, narrow the claim'; < 0.463 => 'worse, claim safe'.
Run: source ~/.theone_keys.env && python experiments/baseline_b_crossbase/run.py
"""
from __future__ import annotations
import importlib.util
import json
import os
import re
import time
import urllib.request
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location(
    "scale_run", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec)
import sys as _sys
_argv = _sys.argv; _sys.argv = ["x"]
spec.loader.exec_module(scale)
_sys.argv = _argv

PROTO = scale.PROTO
SYS_A = scale.SYS_A
_ANS = scale._ANS
TOL = 0.005
N, NODES, BASE_SEED = 150, 12, 20260614


def gen_l_tier():
    rng = np.random.default_rng(BASE_SEED + NODES)
    out = []
    for i in range(N):
        d = scale.gen_dag(NODES, rng)
        g = scale.build_graph(d)
        truth = scale.InterventionEngine(g).query_intervention(
            d["y"], 1, {d["x"]: 1}).value
        out.append({"i": i, "text": scale.render_text(d), "truth": round(truth, 6)})
    return out


def _post(url, headers, body, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def ask_openai(text):
    t0 = time.time()
    try:
        out = _post("https://api.openai.com/v1/chat/completions",
                    {"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    {"model": "gpt-5.1",
                     "messages": [{"role": "system", "content": SYS_A},
                                  {"role": "user", "content": text + PROTO}],
                     "max_completion_tokens": 4096})
        content = out["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 2),
                "fail": f"transport: {e}"[:120]}
    m = None
    for m in _ANS.finditer(content):
        pass
    return {"pred": float(m.group(1)) if m else None,
            "latency": round(time.time() - t0, 2),
            "fail": None if m else "no ANSWER line"}


def ask_gemini(text):
    t0 = time.time()
    try:
        out = _post(f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"gemini-3.5-flash:generateContent?key={os.environ['GEMINI_API_KEY']}",
                    {"Content-Type": "application/json"},
                    {"systemInstruction": {"parts": [{"text": SYS_A}]},
                     "contents": [{"parts": [{"text": text + PROTO}]}],
                     "generationConfig": {"temperature": 0,
                                          "maxOutputTokens": 4096}})
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        content = "".join(p.get("text", "") for p in parts)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 2),
                "fail": f"transport: {e}"[:120]}
    m = None
    for m in _ANS.finditer(content):
        pass
    return {"pred": float(m.group(1)) if m else None,
            "latency": round(time.time() - t0, 2),
            "fail": None if m else "no ANSWER line"}


def main():
    problems = gen_l_tier()
    jpath = HERE / "rows.jsonl"
    done = {}
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done[r["i"]] = r
    jsonl = jpath.open("a")
    rows = []
    for p in problems:
        if p["i"] in done:
            rows.append(done[p["i"]]); continue
        g5 = ask_openai(p["text"])
        gm = ask_gemini(p["text"])
        row = {"i": p["i"], "truth": p["truth"], "gpt51": g5, "gemini35f": gm}
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
        print(f"[X{p['i']:03d}] truth={p['truth']:.4f}  "
              f"gpt5.1={g5['pred']} ({g5['latency']}s)  "
              f"gem3.5f={gm['pred']} ({gm['latency']}s)", flush=True)

    summary = {"n": len(rows), "deepseek_formal_L_acc": 0.613}
    for key in ("gpt51", "gemini35f"):
        ok = sum(1 for r in rows if r[key]["pred"] is not None
                 and abs(r[key]["pred"] - r["truth"]) <= TOL)
        fails = sum(1 for r in rows if r[key]["pred"] is None)
        acc = ok / len(rows)
        summary[key] = {"acc": round(acc, 3), "protocol_failures": fails,
                        "verdict": ("consistent" if abs(acc - 0.613) <= 0.15 else
                                    "base_dependent_better" if acc > 0.763 else
                                    "worse_claim_safe")}
    (HERE / "results.json").write_text(json.dumps({"summary": summary,
                                                   "rows": rows}, indent=2))
    print("\n===== CROSSBASE SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
