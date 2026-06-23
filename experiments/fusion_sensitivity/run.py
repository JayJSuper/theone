"""Fusion deepening④ · latent-confounding sensitivity (E-value) on the do() credential.

Makes the 'latent confounding UNCERTIFIED' limit QUANTITATIVE and recomputable: every
do() now carries an E-value = the minimum strength an unmeasured confounder would need
(risk-ratio scale, with BOTH treatment and outcome) to fully explain away the effect.
  • strong effect -> large E-value -> robust (only an implausible hidden confounder overturns it)
  • weak effect   -> E-value near 1 -> fragile (a mild hidden confounder could overturn it)

Run:  .venv/bin/python experiments/fusion_sensitivity/run.py
"""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.layer2_world_model import CausalLayer, e_value_for_do


def chain(p1, p0):
    """X -> Y with P(Y=1|X=1)=p1, P(Y=1|X=0)=p0 (no modeled confounder)."""
    g = CausalGraph()
    for n in ("X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("X", "Y")
    g.set_cpt("X", {(): {0: 0.5, 1: 0.5}})
    g.set_cpt("Y", {(0,): {0: round(1 - p0, 3), 1: p0}, (1,): {0: round(1 - p1, 3), 1: p1}})
    return g


def main():
    print("=== Fusion deepening④: latent-confounding sensitivity (E-value) ===\n")
    L = CausalLayer()
    ok = True
    rows = [("strong  p1=.8 p0=.2", .8, .2), ("moderate p1=.65 p0=.35", .65, .35),
            ("weak    p1=.55 p0=.45", .55, .45)]
    evals = {}
    print(f"{'effect':<24} {'do(X=1)':>8} {'do(X=0)':>8} {'risk_ratio':>11} {'E-value':>8} {'verdict':>8}")
    for label, p1, p0 in rows:
        v = L.run({"graph": chain(p1, p0), "treatment": "X", "target": "Y"})
        s = v.credential.evidence["sensitivity"]
        evals[label] = s["e_value"]
        print(f"{label:<24} {v.credential.value:>8.3f} "
              f"{v.credential.evidence['do_x0']:>8.3f} {s['risk_ratio']:>11.3f} "
              f"{s['e_value']:>8.2f} {'ANSWER' if v.is_answer() else 'ABSTAIN':>8}")
        ok &= v.is_answer()

    # monotone: stronger effect -> larger E-value (more robust to hidden confounding)
    mono = (evals["strong  p1=.8 p0=.2"] > evals["moderate p1=.65 p0=.35"]
            > evals["weak    p1=.55 p0=.45"])
    ok &= mono
    # independent recompute of the E-value formula
    chk = e_value_for_do(0.8, 0.2)["e_value"]
    ok &= abs(chk - evals["strong  p1=.8 p0=.2"]) < 1e-9

    print("\nReading: the E-value rides on every do() credential as a recomputable bound on the")
    print("ONE thing the engine cannot certify — unmeasured confounding. A strong effect needs")
    print("an implausibly strong hidden confounder to overturn (robust); a weak effect is fragile.")
    print("This turns 'latent-confounding-uncertified' from a warning into a quantitative handle.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}  (E-value monotone in effect size: {mono})")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
