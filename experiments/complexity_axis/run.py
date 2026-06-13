"""Causal complexity axis (frozen AM-011). Fixed structure, scan k = adjustment-
set cardinality (independent confounders), find each base's combinatorial cliff.
x-axis published as 2^k (marginalization combos). Pure-backdoor structure only.
Subjects: gpt-5.1 (flagship) | deepseek-v4-flash (flash) | C (engine).
Scoring: AM-007 (protocol fail = error, no retry). Crash-safe resume.
Run: source ~/.theone_keys.env && python experiments/complexity_axis/run.py
"""
from __future__ import annotations
import itertools
import json
import os
import re
import time
import urllib.request
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
KS = [1, 2, 3, 4, 5, 6]
N_PER_K = 50
BASE_SEED = 20260617
TOL = 0.005
MAXTOK = 4096


def k_graph(k, seed):
    """k independent confounders U0..U{k-1}, each -> X and -> Y; X -> Y; +1
    irrelevant variable W. Adjustment set = all U's (size k). Pure backdoor."""
    rng = np.random.default_rng(seed)
    g = CausalGraph()
    Us = [f"U{i}" for i in range(k)]
    names = Us + ["X", "Y", "W"]
    for n in names:
        g.add_variable(Variable(n))
    for u in Us:
        g.add_edge(u, "X"); g.add_edge(u, "Y")
    g.add_edge("X", "Y")
    pw = round(float(rng.uniform(.3, .7)), 2)
    g.set_cpt("W", {(): {1: pw, 0: 1 - pw}})
    for u in Us:
        p = round(float(rng.uniform(.3, .7)), 2)
        g.set_cpt(u, {(): {1: p, 0: 1 - p}})
    for v in ("X", "Y"):
        order = list(g.parent_order(v))
        rows = {}
        for c in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(.1, .9)), 3)
            rows[c] = {1: p, 0: 1 - p}
        g.set_cpt(v, rows)
    return g


def render(g, k):
    Us = [f"U{i}" for i in range(k)]
    lines = [f"A system has binary variables: {', '.join(Us)}, X, Y, W."]
    for u in Us:
        lines.append(f"{u} directly influences both X and Y.")
    lines.append("X directly influences Y. W influences nothing.")
    lines.append("Measured quantities:")
    for v in g.variables:
        ps = list(g.parent_order(v))
        for combo, d in g.cpt(v).items():
            cond = ",".join(f"{p}={c}" for p, c in zip(ps, combo))
            lines.append(f"P({v}=1|{cond})={d[1]:.3f}" if ps
                         else f"P({v}=1)={d[1]:.3f}")
    lines.append("Question: what is P(Y=1|do(X=1)), the probability of Y=1 if X "
                 "were set to 1 by intervention? Give 4 decimals.")
    return "\n".join(lines)


PROTO = "\nEnd your reply with exactly one line:\nANSWER: <number with 4 decimals>"
SYS = ("You are an expert in causal inference. For P(Y|do(X=x)), use the backdoor "
       "adjustment over ALL confounders: P(Y=1|do(X=1)) = sum over all confounder "
       "configurations u of P(Y=1|X=1,u) * P(u). Work carefully.")
_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)


def _last(content):
    m = None
    for m in _ANS.finditer(content or ""):
        pass
    return float(m.group(1)) if m else None


def ask_openai(text):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": text + PROTO}],
                "max_completion_tokens": MAXTOK}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]
        tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1),
                "tokens": 0, "fail": str(e)[:80]}
    return {"pred": _last(c), "latency": round(time.time() - t0, 1),
            "tokens": tok, "fail": None if _last(c) is not None else "no ANSWER"}


def ask_deepseek(text):
    from theone.llm import DeepSeekClient
    t0 = time.time()
    try:
        out = DeepSeekClient(timeout=200).chat(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": text + PROTO}], max_tokens=MAXTOK,
            temperature=0.0)
        c, tok = out["content"], out["usage"].get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1),
                "tokens": 0, "fail": str(e)[:80]}
    return {"pred": _last(c), "latency": round(time.time() - t0, 1),
            "tokens": tok, "fail": None if _last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "rows.jsonl"
    done = {}
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done[(r["k"], r["i"])] = r
    jsonl = jpath.open("a")
    rows = []
    for k in KS:
        for i in range(N_PER_K):
            g = k_graph(k, BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            if (k, i) in done:
                rows.append(done[(k, i)]); continue
            text = render(g, k)
            gpt = ask_openai(text)
            ds = ask_deepseek(text)
            row = {"k": k, "i": i, "combos": 2 ** k, "truth": truth,
                   "gpt51": gpt, "deepseek": ds, "C": truth}  # C = engine = exact
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
            print(f"[k{k}-{i:02d}] 2^k={2**k:>3} truth={truth:.4f}  "
                  f"gpt5.1={gpt['pred']}({gpt['latency']}s)  "
                  f"ds={ds['pred']}({ds['latency']}s)", flush=True)

    summary = {"per_k": {}}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]
        def acc(key):
            return round(sum(1 for r in kr if r[key]["pred"] is not None
                             and abs(r[key]["pred"] - r["truth"]) <= TOL) / len(kr), 3) \
                if key != "C" else 1.0
        summary["per_k"][k] = {
            "combos": 2 ** k,
            "gpt51_acc": acc("gpt51"),
            "deepseek_acc": acc("deepseek"),
            "C_acc": 1.0,
            "gpt51_fail": sum(1 for r in kr if r["gpt51"]["pred"] is None),
            "deepseek_fail": sum(1 for r in kr if r["deepseek"]["pred"] is None)}
    # cliff = first k where acc < 0.85
    def cliff(key):
        for k in KS:
            if summary["per_k"][k][f"{key}_acc"] < 0.85:
                return k
        return None
    summary["cliff"] = {"gpt51": cliff("gpt51"), "deepseek": cliff("deepseek")}
    (HERE / "results.json").write_text(json.dumps(
        {"summary": summary, "rows": rows}, indent=2))
    print("\n===== COMPLEXITY AXIS SUMMARY =====")
    print(f"{'k':>3}{'2^k':>5}{'gpt5.1':>9}{'deepseek':>10}{'C':>6}")
    for k in KS:
        s = summary["per_k"][k]
        print(f"{k:>3}{s['combos']:>5}{s['gpt51_acc']:>9}{s['deepseek_acc']:>10}{1.0:>6}")
    print(f"cliff (first k<0.85): {summary['cliff']}")


if __name__ == "__main__":
    main()
