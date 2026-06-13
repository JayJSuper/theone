"""Bet ① ecological validity, step 3: WRONG-STRUCTURE robustness (no API, pure
computation). The engine + credential guarantee "exact given the structure" — so
what happens when the adjustment set is WRONG (a confounder is missed)? This tests
the LIMIT of what a cognitive credential certifies, and whether the engine still
beats the collapsed LLM under imperfect structure.

Reuse the frozen cliff SCMs (k independent confounders U0..U{k-1}). Back-door
adjustment over a subset S of confounders:
    do_S(X=1) = Σ_s [Π_{i∈S} P(U_i=s_i)] · P(Y=1 | X=1, S=s)
  S = all U's  -> correct do (= engine.query_intervention, sanity-checked)
  S = all but one -> MISSING-confounder bias (realistic structural error)
  S = {} -> no adjustment = the naive (fully confounded) observational estimate

Honest framing: a credential certifies the COMPUTATION, not the structure's
correctness. A missed confounder yields a confidently-wrong answer with a valid
computation credential. We quantify that bias vs k and vs confounder strength,
and compare it to (a) full confounding (no adjustment) and (b) the LLM's high-k
reasoning collapse (from the cliff). Connects to NOTE-002 (silent bias).

Run: python experiments/wrong_structure/run.py
"""
from __future__ import annotations
import importlib.util, itertools, json
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

KS = [2, 3, 4, 5, 6]
N = 80


def do_subset(g, eng, k, S):
    """Back-door adjustment over confounder subset S (list of U-names)."""
    Us = [f"U{i}" for i in range(k)]
    pU = {u: g.cpt(u)[()][1] for u in Us}
    total = 0.0
    for combo in itertools.product([0, 1], repeat=len(S)):
        s = dict(zip(S, combo))
        pS = 1.0
        for u in S:
            pS *= pU[u] if s[u] == 1 else (1 - pU[u])
        pY = eng.query_observation("Y", 1, {"X": 1, **s}).value
        total += pS * pY
    return total


def main():
    rows = []
    for k in KS:
        Us = [f"U{i}" for i in range(k)]
        for i in range(N):
            g = R.k_graph(k, R.BASE_SEED + 1000 * k + i)
            eng = InterventionEngine(g)
            true_do = eng.query_intervention("Y", 1, {"X": 1}).value
            # sanity: subset=all reproduces true_do
            full = do_subset(g, eng, k, Us)
            assert abs(full - true_do) < 1e-9, f"subset-all != true_do at k{k}i{i}"
            no_adj = do_subset(g, eng, k, [])            # naive observational P(Y=1|X=1)
            # missing each single confounder; record worst and mean bias
            miss_bias = []
            for j in range(k):
                S = [u for u in Us if u != Us[j]]
                miss_bias.append(abs(do_subset(g, eng, k, S) - true_do))
            rows.append({"k": k, "i": i, "true_do": round(true_do, 6),
                         "no_adjust_bias": round(abs(no_adj - true_do), 6),
                         "miss1_bias_mean": round(float(np.mean(miss_bias)), 6),
                         "miss1_bias_worst": round(float(np.max(miss_bias)), 6)})
    # summary
    print(f"{'k':>3} | {'no-adjust bias':>15} | {'miss-1 mean':>12} | {'miss-1 worst':>12}")
    summ = {}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]
        na = np.mean([r["no_adjust_bias"] for r in kr])
        mm = np.mean([r["miss1_bias_mean"] for r in kr])
        mw = np.mean([r["miss1_bias_worst"] for r in kr])
        summ[k] = {"no_adjust_bias": round(float(na), 4), "miss1_mean": round(float(mm), 4),
                   "miss1_worst": round(float(mw), 4)}
        print(f"{k:>3} | {na:>15.4f} | {mm:>12.4f} | {mw:>12.4f}")
    (HERE / "results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
    print("\nReading: 'no-adjust' = full confounding (worst case, what you get with NO causal model).")
    print("'miss-1' = drop ONE confounder from the adjustment set (a realistic structural error).")
    print("Compare to LLM high-k reasoning error ~0.05-0.10 (cliff/CPT) to see if partial structure still beats collapse.")


if __name__ == "__main__":
    main()
