"""Baseline B v0 — frozen prereg 8a11580. Run:
  source ~/.theone_keys.env && python experiments/baseline_b/run.py
A = raw LLM | B = LLM + backdoor scaffold | C = The One (parser -> S2 engine).
All criteria frozen; results published as-is.
"""
from __future__ import annotations
import json
import math
import re
import time
import hashlib
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.llm import DeepSeekClient

HERE = Path(__file__).parent
SEED = 20260612
N_IDENT, N_UNIDENT = 30, 5

# ---------------------------------------------------------------- generator
def make_graph(p_u, px1, px0, py):
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: p_u, 0: 1 - p_u}})
    g.set_cpt("X", {(1,): {1: px1, 0: 1 - px1}, (0,): {1: px0, 0: 1 - px0}})
    order = list(g.parent_order("Y"))
    cpt = {}
    for (uv, xv), p1 in py.items():
        key = tuple(uv if p == "U" else xv for p in order)
        cpt[key] = {1: p1, 0: 1 - p1}
    g.set_cpt("Y", cpt)
    return g


def gen_problems():
    rng = np.random.default_rng(SEED)
    problems = []
    for i in range(N_IDENT + N_UNIDENT):
        p_u = round(float(rng.uniform(0.3, 0.7)), 2)
        while True:
            px1 = round(float(rng.uniform(0.1, 0.9)), 2)
            px0 = round(float(rng.uniform(0.1, 0.9)), 2)
            if abs(px1 - px0) >= 0.2:
                break
        py = {(1, 1): round(float(rng.uniform(0.05, 0.95)), 2),
              (0, 1): round(float(rng.uniform(0.05, 0.95)), 2),
              (1, 0): round(float(rng.uniform(0.05, 0.95)), 2),
              (0, 0): round(float(rng.uniform(0.05, 0.95)), 2)}
        g = make_graph(p_u, px1, px0, py)
        eng = InterventionEngine(g)
        truth = eng.query_intervention("Y", 1, {"X": 1}).value
        ident = i < N_IDENT
        if ident:
            text = (
                f"A study involves a binary risk factor U, a binary exposure X and a "
                f"binary outcome Y. U influences both X and Y; X influences Y.\n"
                f"Measured quantities: P(U=1)={p_u}. "
                f"P(X=1|U=1)={px1}, P(X=1|U=0)={px0}. "
                f"P(Y=1|X=1,U=1)={py[(1,1)]}, P(Y=1|X=1,U=0)={py[(0,1)]}, "
                f"P(Y=1|X=0,U=1)={py[(1,0)]}, P(Y=1|X=0,U=0)={py[(0,0)]}.\n"
                f"Question: what is P(Y=1|do(X=1)), the probability of Y=1 if X were "
                f"set to 1 by intervention? Give 4 decimals.")
        else:
            # observational P(Y=1|X=1) computable by us, shown to subjects
            obs1 = eng.query_observation("Y", 1, {"X": 1}).value
            obs0 = eng.query_observation("Y", 1, {"X": 0}).value
            text = (
                f"A study involves a binary exposure X and a binary outcome Y. A "
                f"confounder U is known to influence both X and Y, but U WAS NOT "
                f"MEASURED — no probabilities involving U are available.\n"
                f"Measured quantities (observational only): "
                f"P(Y=1|X=1)={obs1:.4f}, P(Y=1|X=0)={obs0:.4f}.\n"
                f"Question: what is P(Y=1|do(X=1))? Give 4 decimals, or state that "
                f"it cannot be determined.")
        problems.append({"id": i, "identifiable": ident, "text": text,
                         "truth": round(truth, 6),
                         "params": {"p_u": p_u, "px1": px1, "px0": px0,
                                    "py": {str(k): v for k, v in py.items()}}})
    return problems


# ---------------------------------------------------------------- subjects
PROTO = ("\nEnd your reply with exactly one line:\n"
         "ANSWER: <number with 4 decimals>   or   ANSWER: CANNOT_DETERMINE")

SYS_A = "You answer probability questions. Be precise."
SYS_B = ("You are an expert in causal inference (Pearl framework). For intervention "
         "questions P(Y|do(X)), beware confounding: the observational P(Y|X) is "
         "biased when a common cause exists. If the confounder U is measured, use "
         "the backdoor adjustment: P(Y=1|do(X=1)) = sum_u P(Y=1|X=1,U=u) * P(U=u). "
         "If the needed adjustment variable is NOT measured, the effect is not "
         "identifiable. Work step by step, do the arithmetic carefully.")

_ANS = re.compile(r"ANSWER:\s*(CANNOT_DETERMINE|[0-9.]+)", re.I)


def ask_llm(client, system, text, max_tokens):
    out = client.chat([{"role": "system", "content": system},
                       {"role": "user", "content": text + PROTO}],
                      max_tokens=max_tokens, temperature=0.0)
    m = None
    for m in _ANS.finditer(out["content"] or ""):
        pass
    if not m:
        return {"raw": (out["content"] or "")[-200:], "parse": None}
    v = m.group(1).upper()
    return {"parse": "ABSTAIN" if v == "CANNOT_DETERMINE" else float(v)}


def the_one(problem):
    """C: deterministic parser (format-aware, honest scope) -> S2 engine."""
    t = problem["text"]
    if "WAS NOT MEASURED" in t:
        return "ABSTAIN"                      # identifiability gate
    g = re.search
    p_u = float(g(r"P\(U=1\)=([0-9]*\.?[0-9]+)", t).group(1))
    px1 = float(g(r"P\(X=1\|U=1\)=([0-9]*\.?[0-9]+)", t).group(1))
    px0 = float(g(r"P\(X=1\|U=0\)=([0-9]*\.?[0-9]+)", t).group(1))
    py = {(1, 1): float(g(r"P\(Y=1\|X=1,U=1\)=([0-9]*\.?[0-9]+)", t).group(1)),
          (0, 1): float(g(r"P\(Y=1\|X=1,U=0\)=([0-9]*\.?[0-9]+)", t).group(1)),
          (1, 0): float(g(r"P\(Y=1\|X=0,U=1\)=([0-9]*\.?[0-9]+)", t).group(1)),
          (0, 0): float(g(r"P\(Y=1\|X=0,U=0\)=([0-9]*\.?[0-9]+)", t).group(1))}
    eng = InterventionEngine(make_graph(p_u, px1, px0, py))
    return round(eng.query_intervention("Y", 1, {"X": 1}).value, 6)


# ---------------------------------------------------------------- run
def main():
    t0 = time.time()
    client = DeepSeekClient()
    problems = gen_problems()
    rows = []
    for p in problems:
        a = ask_llm(client, SYS_A, p["text"], 1024)
        b = ask_llm(client, SYS_B, p["text"], 2048)
        c = the_one(p)
        rows.append({**{k: p[k] for k in ("id", "identifiable", "truth")},
                     "A": a.get("parse"), "B": b.get("parse"), "C": c})
        print(f"#{p['id']:>2} ident={p['identifiable']} truth={p['truth']:.4f}  "
              f"A={a.get('parse')}  B={b.get('parse')}  C={c}")
    elapsed = time.time() - t0

    ident = [r for r in rows if r["identifiable"]]
    unid = [r for r in rows if not r["identifiable"]]

    def err(v, truth):
        if v is None or v == "ABSTAIN":
            return None
        return abs(float(v) - truth)

    res = {"n_ident": len(ident), "n_unident": len(unid), "elapsed_s": round(elapsed, 1)}
    for s in ("A", "B", "C"):
        errs = [err(r[s], r["truth"]) for r in ident]
        valid = [e for e in errs if e is not None]
        res[f"{s}_protocol_failures"] = sum(1 for r in ident if r[s] is None)
        res[f"{s}_false_abstain_ident"] = sum(1 for r in ident if r[s] == "ABSTAIN")
        res[f"{s}_mae"] = round(float(np.mean(valid)), 6) if valid else None
        res[f"{s}_abstain_unident"] = sum(1 for r in unid if r[s] == "ABSTAIN")
        res[f"{s}_fabricated_unident"] = sum(
            1 for r in unid if isinstance(r[s], float))
    # paired sign tests C vs A / B (strict wins, ties dropped)
    from math import comb
    for opp in ("A", "B"):
        wins = ties = n = 0
        for r in ident:
            ec, eo = err(r["C"], r["truth"]), err(r[opp], r["truth"])
            if ec is None or eo is None:
                continue
            n += 1
            if ec < eo: wins += 1
            elif ec == eo: ties += 1
        eff = n - ties
        p_val = (sum(comb(eff, k) for k in range(wins, eff + 1)) / 2**eff
                 if eff else 1.0)
        res[f"C_vs_{opp}"] = {"wins": wins, "ties": ties, "n": n,
                              "sign_p_one_sided": round(p_val, 6)}
    # frozen verdict
    ok_mae = (res["C_mae"] is not None and res["A_mae"] is not None and
              res["B_mae"] is not None and
              res["C_mae"] < res["A_mae"] and res["C_mae"] < res["B_mae"])
    ok_sign = all(res[f"C_vs_{o}"]["sign_p_one_sided"] < 0.05 for o in ("A", "B"))
    ok_abst = res["C_abstain_unident"] == len(unid)
    res["frozen_verdict_C_wins"] = bool(ok_mae and ok_sign and ok_abst)
    res["verdict_parts"] = {"mae": ok_mae, "sign": ok_sign, "abstention": ok_abst}

    (HERE / "results.json").write_text(json.dumps(
        {"summary": res, "rows": rows}, indent=2))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()[:16]
    print("\n================ SUMMARY ================")
    for s in ("A", "B", "C"):
        print(f"{s}: MAE={res[f'{s}_mae']}  protocol_fail={res[f'{s}_protocol_failures']}  "
              f"abstain_on_unident={res[f'{s}_abstain_unident']}/{len(unid)}  "
              f"fabricated_on_unident={res[f'{s}_fabricated_unident']}")
    print(f"C vs A: {res['C_vs_A']}")
    print(f"C vs B: {res['C_vs_B']}")
    print(f"FROZEN VERDICT — C wins: {res['frozen_verdict_C_wins']}  parts={res['verdict_parts']}")
    print(f"elapsed {elapsed:.0f}s  results_sha={sha}")


if __name__ == "__main__":
    main()
