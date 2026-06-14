"""Financial CPT-misspecification stress test (Jack Q-C29 §1.1): in finance the CPT
uncertainty is STRUCTURAL, not just statistical — tail dependence spikes in crises,
which are rare in the calibration history. This is the 2008 Gaussian-copula failure
(Li's model used normal-times correlation). The engine computes do() EXACTLY on
whatever CPT it is given, so a normal-times-calibrated CPT yields a confidently
wrong stressed risk — and the credential certifies the computation, NOT the
calibration (the financial form of NOTE-004; the SR 11-7 'model risk' the credential
does not catch).

We compare, on identical structure, do(distress=1) computed on:
  - TRUE CPT (crisis tail amplification amp_true)
  - MISCALIBRATED CPT (normal-times amp_mis < amp_true)
for two measures:
  - unconditional P(Y=1|do(D=1))                       (diluted: crisis configs rare)
  - STRESSED P(Y=1|do(D=1), >= ceil(k/2) factors firing)  (what risk mgmt cares about)
No API — pure engine computation. The point is the engine's dependence on CPT
calibration, not the LLM comparison.

Run: python experiments/finance_cpt_misspec/run.py
"""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
KS = [4, 5, 6]
N = 50


def credit_scm(k, seed, amp):
    rng = np.random.default_rng(seed); g = CausalGraph()
    Ss = [f"S{i}" for i in range(k)]
    for n in Ss + ["D", "Y"]:
        g.add_variable(Variable(n))
    for s in Ss:
        g.add_edge(s, "D"); g.add_edge(s, "Y")
    g.add_edge("D", "Y")
    pS = {}
    for s in Ss:
        p = round(float(rng.uniform(.03, .20)), 3); pS[s] = p; g.set_cpt(s, {(): {1: p, 0: round(1 - p, 3)}})
    oD = list(g.parent_order("D")); rows = {}
    for c in itertools.product((1, 0), repeat=len(oD)):
        nf = sum(c); p = round(min(.95, max(.05, 0.2 + 0.5 * nf / k)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("D", rows)
    oY = list(g.parent_order("Y")); rows = {}
    for c in itertools.product((1, 0), repeat=len(oY)):
        d = c[oY.index("D")]; nf = sum(c[i] for i, v in enumerate(oY) if v != "D")
        p = round(min(.98, max(.02, 0.10 + 0.25 * d + amp * nf / k)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("Y", rows)
    return g, pS


def do_measures(g, pS, k):
    """do(D=1): unconditional P(Y=1) and stressed P(Y=1 | >= ceil(k/2) factors firing)."""
    Ss = [f"S{i}" for i in range(k)]
    oY = list(g.parent_order("Y"))
    thr = math.ceil(k / 2)
    num_u = den_u = num_s = den_s = 0.0
    for combo in itertools.product((0, 1), repeat=k):
        w = 1.0
        for i, s in enumerate(Ss):
            w *= pS[s] if combo[i] == 1 else (1 - pS[s])
        key = tuple(1 if v == "D" else combo[Ss.index(v)] for v in oY)  # D set to 1
        py = g.cpt("Y")[key][1]
        num_u += w * py; den_u += w
        if sum(combo) >= thr:
            num_s += w * py; den_s += w
    return num_u / den_u, (num_s / den_s if den_s > 0 else float("nan"))


def main():
    rows = []
    for k in KS:
        for i in range(N):
            seed = 9000 + 100 * k + i
            gt, pS = credit_scm(k, seed, amp=0.6)        # true crisis amplification
            gm, _ = credit_scm(k, seed, amp=0.15)        # normal-times (miscalibrated)
            ut, st = do_measures(gt, pS, k)
            um, sm = do_measures(gm, pS, k)
            rows.append({"k": k, "i": i, "uncond_true": ut, "uncond_mis": um,
                         "stressed_true": st, "stressed_mis": sm,
                         "uncond_bias": um - ut, "stressed_bias": sm - st})
    print("Engine computes do() EXACTLY on whatever CPT given. Normal-times calibration")
    print("(amp 0.15) vs true crisis amplification (amp 0.6). Bias = miscal - true (signed):\n")
    print(f"{'k':>2} | {'uncond bias':>12} | {'STRESSED bias':>13} | {'stressed true->mis':>20}")
    summ = {}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]
        ub = np.mean([r["uncond_bias"] for r in kr]); sb = np.mean([r["stressed_bias"] for r in kr])
        st_t = np.mean([r["stressed_true"] for r in kr]); st_m = np.mean([r["stressed_mis"] for r in kr])
        summ[k] = {"uncond_bias": round(float(ub), 4), "stressed_bias": round(float(sb), 4),
                   "stressed_true": round(float(st_t), 3), "stressed_mis": round(float(st_m), 3)}
        print(f"{k:>2} | {ub:>+12.4f} | {sb:>+13.4f} | {st_t:.3f} -> {st_m:.3f}")
    (HERE / "results.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
    print("\nReading: unconditional bias is diluted (crisis configs are rare). The STRESSED")
    print("measure -- the one risk management acts on -- carries the large silent bias.")
    print("The engine's credential certifies the computation is exact; it does NOT certify")
    print("the CPT reflects crisis conditions. This is the 2008 copula failure mode, and")
    print("the financial form of NOTE-004 (credential != structural/calibration correctness).")


if __name__ == "__main__":
    main()
