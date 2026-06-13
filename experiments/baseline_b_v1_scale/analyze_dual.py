"""Dual-track final analysis: ecological (random DAG) vs clean (fixed skeleton).
Run after both formal runs complete:  python .../analyze_dual.py
Answers: is the LLM collapse driven by causal complexity or by node count?
"""
from __future__ import annotations
import json
from math import comb
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TOL = 0.005


def load(path):
    return [json.loads(l) for l in (HERE / path).read_text().splitlines() if l.strip()]


def correct(r, s):
    p = r[s]["pred"]
    return p is not None and abs(p - r["truth"]) <= TOL


def tier_stats(rows, tier):
    tr = [r for r in rows if r["tier"] == tier]
    out = {}
    for s in "ABC":
        out[s] = {"acc": round(sum(correct(r, s) for r in tr) / len(tr), 3),
                  "fail": sum(1 for r in tr if r[s]["pred"] is None),
                  "lat": round(float(np.mean([r[s]["latency"] for r in tr])), 1)}
    # sign test C vs stronger opponent
    opp = "A" if out["A"]["acc"] >= out["B"]["acc"] else "B"
    wins = sum(1 for r in tr if correct(r, "C") and not correct(r, opp))
    losses = sum(1 for r in tr if not correct(r, "C") and correct(r, opp))
    n = wins + losses
    out["sep"] = round(out["C"]["acc"] - max(out["A"]["acc"], out["B"]["acc"]), 3)
    out["sign"] = {"wins": wins, "losses": losses,
                   "p": (sum(comb(n, k) for k in range(wins, n + 1)) / 2**n
                         if n else 1.0)}
    return out


def main():
    eco = load("rows.random.jsonl")
    clean = load("rows.fixed.jsonl")
    print(f"ecological n={len(eco)}  clean n={len(clean)}")
    print(f"\n{'':>6}{'— 生态版(随机DAG) —':^36}{'— 干净版(固定骨架) —':^36}")
    print(f"{'tier':>6}{'A':>8}{'B':>8}{'C':>8}{'分离':>9}"
          f"{'A':>10}{'B':>8}{'C':>8}{'分离':>9}")
    summary = {}
    for tier in ("S", "M", "L"):
        e, c = tier_stats(eco, tier), tier_stats(clean, tier)
        summary[tier] = {"eco": e, "clean": c}
        print(f"{tier:>6}{e['A']['acc']:>8}{e['B']['acc']:>8}{e['C']['acc']:>8}"
              f"{e['sep']:>+9}{c['A']['acc']:>10}{c['B']['acc']:>8}"
              f"{c['C']['acc']:>8}{c['sep']:>+9}")
    # the interpretive question
    eL, cL = summary["L"]["eco"], summary["L"]["clean"]
    if cL["A"]["acc"] >= 0.9 and eL["A"]["acc"] <= 0.7:
        verdict = ("COMPLEXITY-DRIVEN: with identification difficulty held constant "
                   "(single backdoor + distractors), LLMs stay accurate at 12 nodes; "
                   "the ecological collapse is driven by CAUSAL STRUCTURE COMPLEXITY, "
                   "not node count per se.")
    elif cL["A"]["acc"] <= 0.7:
        verdict = ("SCALE-DRIVEN: LLMs collapse even with constant identification "
                   "difficulty - raw graph size alone breaks the reasoning chain.")
    else:
        verdict = "MIXED: partial degradation in both; report both curves."
    print(f"\nINTERPRETATION: {verdict}")
    (HERE / "dual_track_summary.json").write_text(json.dumps(
        {"summary": summary, "interpretation": verdict}, indent=2))


if __name__ == "__main__":
    main()
