"""Fusion deepening③ · L3 cognitive updater — propose a structure update only when
new data justifies it by a recomputable BIC improvement.

Old held model: X -> Y (Z believed irrelevant). Nodes are {X, Z, Y} throughout.
  A. new data from a SHIFTED regime (Z -> Y, X now irrelevant): re-discovery finds the
     new skeleton and BIC improves far beyond margin -> ANSWER (propose update).
  B. new data from the SAME regime (X -> Y): re-discovery == current model -> ABSTAIN
     (no structural change to propose; do not churn).

Run:  .venv/bin/python experiments/fusion_cognitive_update/run.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.layer3_decision import CognitiveUpdater


def gen_xy(n, seed):
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < 0.5).astype(int)
    z = (rng.random(n) < 0.5).astype(int)            # irrelevant
    y = (rng.random(n) < np.where(x == 1, 0.8, 0.2)).astype(int)
    return pd.DataFrame({"X": x, "Z": z, "Y": y})


def gen_zy(n, seed):
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < 0.5).astype(int)            # now irrelevant
    z = (rng.random(n) < 0.5).astype(int)
    y = (rng.random(n) < np.where(z == 1, 0.8, 0.2)).astype(int)
    return pd.DataFrame({"X": x, "Z": z, "Y": y})


def main():
    print("=== Fusion deepening③: L3 cognitive updater (BIC-gated structure update) ===\n")
    U = CognitiveUpdater()
    old_edges = [("X", "Y")]
    ok = True

    # A. regime shift -> propose update
    va = U.run({"old_edges": old_edges, "data": gen_zy(3000, 0)})
    if va.is_answer():
        ev = va.credential.evidence
        _, info = va.credential.verify()
        print(f"A shifted regime (Z->Y) -> ANSWER (propose update)")
        print(f"   old {ev['old_edges']} -> proposed {ev['proposed_edges']} | "
              f"BIC delta={ev['bic_delta']} (> margin {ev['margin']}) | recompute gap={info.get('gap',0):.1e}")
    a_ok = va.is_answer() and va.credential.value > 10
    ok &= a_ok

    # B. stationary regime -> abstain (no change)
    vb = U.run({"old_edges": old_edges, "data": gen_xy(3000, 1)})
    b_ok = not vb.is_answer()
    print(f"\nB stationary regime (X->Y) -> {'ABSTAIN' if b_ok else 'ANSWER'}: "
          f"{vb.reason if b_ok else 'unexpected update'}")
    ok &= b_ok

    print("\nCognitive-updater contract: a structure change is proposed ONLY when new data")
    print("improves BIC beyond a margin AND the skeleton actually changed; otherwise abstain")
    print("(keep the current model, do not churn on noise). The proposal inherits the L2")
    print("discovery limits (orientation / latent confounding uncertified) in its regime.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
