"""Constraint credentials — extending the auditable-causal-inference standard's
criterion 4 (declared regime/boundary) into *checked* domain/physical constraints.

Fusion of two threads:
  (a) The One's credential philosophy: certify, third-party-recomputably, not just
      the number but its admissibility.
  (b) The external AGI构想's EBM idea: assign infinite energy to *physically
      impossible* trajectories — but done *verifiably* (a deterministic, anyone-can-
      recheck inequality), not as a soft neural penalty.

A constraint credential attaches to every causal-effect output a set of declared,
independently-recomputable admissibility checks:
  • BOUNDS:   a probability must lie in [0,1].   (LLMs violate this — see §6.3:
              gpt-5.1 emitted P=1.0179, 1.28 at the cliff, unflagged.)
  • MONOTONE: if a domain declares the effect sign (e.g. distress raises default),
              do(X=1) must be ≥ do(X=0).         (A misspecified CPT can flip it.)
  • NORMALISED: a stated probability and its complement must sum to 1.

The engine's exact outputs pass by construction; the credential's job is to make
*violations* — whether from an LLM's impossible value or a misspecified structure —
machine-evident, so the metacognitive layer can ABSTAIN rather than emit them.

Run:  .venv/bin/python experiments/constraint_credential/run.py
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
TOL = 1e-9


def constraint_credential(value, *, do0=None, declared_sign=None, complement=None):
    """Deterministic, third-party-recomputable admissibility checks on a causal output.
    Returns {check: PASS/VIOLATED/NA}. Anyone can re-verify each inequality by hand."""
    c = {}
    c["bounds_0_1"] = "PASS" if (0.0 - TOL <= value <= 1.0 + TOL) else "VIOLATED"
    if declared_sign is not None and do0 is not None:
        gap = value - do0
        ok = (gap >= -TOL) if declared_sign == "positive" else (gap <= TOL)
        c["monotone_" + declared_sign] = "PASS" if ok else "VIOLATED"
    if complement is not None:
        c["normalised"] = "PASS" if abs((value + complement) - 1.0) <= 1e-6 else "VIOLATED"
    return c


def verdict(checks):
    return "ADMISSIBLE" if all(v == "PASS" for v in checks.values() if v != "NA") else "INADMISSIBLE → ABSTAIN/flag"


def confounded(k, seed, flip=False):
    """k-confounder back-door SCM. If flip, corrupt Y's CPT so the X→Y effect sign
    inverts (a misspecification that a monotonicity constraint should catch)."""
    rng = np.random.default_rng(seed); g = CausalGraph()
    Us = [f"U{i}" for i in range(k)]
    for n in Us + ["X", "Y"]:
        g.add_variable(Variable(n))
    for u in Us:
        g.add_edge(u, "X"); g.add_edge(u, "Y")
    g.add_edge("X", "Y")
    for u in Us:
        p = round(float(rng.uniform(.3, .7)), 3); g.set_cpt(u, {(): {1: p, 0: round(1 - p, 3)}})
    xorder = list(g.parent_order("X")); xrows = {}
    for combo in itertools.product((1, 0), repeat=len(xorder)):
        p = round(float(rng.uniform(.3, .7)), 3); xrows[combo] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("X", xrows)
    order = list(g.parent_order("Y"))
    rows = {}
    for combo in itertools.product((1, 0), repeat=len(order)):
        x_is_1 = combo[order.index("X")] == 1
        base = rng.uniform(.2, .45)
        # positive X→Y effect: X=1 lifts P(Y); flip reverses it (misspecification)
        lift = 0.35 if (x_is_1 != flip) else 0.0
        p = round(min(0.95, base + lift), 3)
        rows[combo] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("Y", rows)
    return g


def main():
    rows = []
    print("=== constraint credentials: making (in)admissibility machine-evident ===\n")

    # (A) engine on a normal SCM with a declared POSITIVE effect — should be admissible
    g = confounded(3, 7)
    eng = InterventionEngine(g)
    do1 = round(eng.query_intervention("Y", 1, {"X": 1}).value, 6)
    do0 = round(eng.query_intervention("Y", 1, {"X": 0}).value, 6)
    comp1 = round(eng.query_intervention("Y", 0, {"X": 1}).value, 6)
    ccA = constraint_credential(do1, do0=do0, declared_sign="positive", complement=comp1)
    rows.append({"case": "engine, normal SCM, declared positive", "value": do1,
                 "do0": do0, "checks": ccA, "verdict": verdict(ccA)})
    print(f"(A) engine normal: do(X=1)={do1} do(X=0)={do0} | {ccA} → {verdict(ccA)}")

    # (B) LLM impossible values from the cliff (§6.3) fed through the SAME credential
    for bad in (1.0179, 1.28):
        ccB = constraint_credential(bad, do0=do0, declared_sign="positive", complement=round(1 - bad, 4))
        rows.append({"case": f"LLM cliff output {bad} (§6.3)", "value": bad,
                     "checks": ccB, "verdict": verdict(ccB)})
        print(f"(B) LLM output {bad}: {ccB} → {verdict(ccB)}  "
              f"[LLM emitted this unflagged; the credential catches it]")

    # (C) engine on a MISSPECIFIED SCM (effect sign flipped) with declared positive
    gf = confounded(3, 7, flip=True)
    engf = InterventionEngine(gf)
    do1f = round(engf.query_intervention("Y", 1, {"X": 1}).value, 6)
    do0f = round(engf.query_intervention("Y", 1, {"X": 0}).value, 6)
    ccC = constraint_credential(do1f, do0=do0f, declared_sign="positive")
    rows.append({"case": "engine, misspecified SCM (sign flipped), declared positive",
                 "value": do1f, "do0": do0f, "checks": ccC, "verdict": verdict(ccC)})
    print(f"(C) engine misspec: do(X=1)={do1f} do(X=0)={do0f} | {ccC} → {verdict(ccC)}  "
          f"[exact computation, but violates the declared domain monotonicity → abstain]")

    (HERE / "results.json").write_text(json.dumps(rows, indent=2))
    print("\nTakeaway: the constraint credential is a *verifiable* EBM — it assigns "
          "'inadmissible' to physically/domain-impossible outputs via deterministic,\n"
          "re-checkable inequalities (not a soft neural penalty). It catches BOTH an "
          "LLM's impossible value (which the LLM emits unflagged) AND a misspecified\n"
          "structure's sign violation (which a bare 'computation-exact' credential would "
          "pass). This extends auditable-causal-inference criterion 4 from *declared* to "
          "*checked* boundaries.")


if __name__ == "__main__":
    main()
