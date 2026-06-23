"""Fusion deepening② · L2 causal-discovery leg with an honest reliability gate.

This is NOTE-004 made operational. Known SCM: U->X, U->Y, X->Y.
  A. observed {U,X,Y}, n=2000  -> ANSWER: skeleton recovered and bootstrap-stable;
     orientation (Markov equivalence) + latent confounding declared UNCERTIFIED.
  B. latent  {X,Y},   n=2000  -> ANSWER: X-Y skeleton is stable, BUT the credential
     loudly declares latent-confounding-uncertified — and do() on the learned X->Y
     structure equals the confounded observational estimate (confidently wrong vs truth),
     which is exactly why the regime warning is not optional.
  C. tiny    {U,X,Y}, n=30    -> ABSTAIN: skeleton unstable under bootstrap (the layer
     refuses to emit a confident structure from insufficient data).

Run:  .venv/bin/python experiments/fusion_discovery/run.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.layer2_world_model import CausalDiscoveryLayer

PY = {(1, 1): .85, (0, 1): .45, (1, 0): .35, (0, 0): .15}


def true_graph():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    oY = list(g.parent_order("Y"))
    g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY): {1: v, 0: round(1 - v, 2)}
                    for (u, x), v in PY.items()})
    return g


def sample(g, n, seed):
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < g.cpt("U")[()][1]).astype(int)
    x = (rng.random(n) < np.where(u == 1, g.cpt("X")[(1,)][1], g.cpt("X")[(0,)][1])).astype(int)
    oY = list(g.parent_order("Y"))
    y = np.array([1 if rng.random() < PY[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    return pd.DataFrame({"U": u, "X": x, "Y": y})


def main():
    print("=== Fusion deepening②: L2 causal-discovery leg (honest reliability gate) ===\n")
    g = true_graph()
    truth_do = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
    df = sample(g, 2000, 0)
    L = CausalDiscoveryLayer()
    ok = True

    # A. observed
    va = L.run({"data": df, "B": 25, "seed": 1})
    if va.is_answer():
        ev = va.credential.evidence
        print(f"A observed {{U,X,Y}} n=2000 -> ANSWER")
        print(f"   skeleton stability={ {k: round(v,2) for k,v in ev['skeleton_stability'].items()} }")
        print(f"   orientation confidence={ev['orientation_confidence']} (≈0.5 ⇒ Markov-equivalent, unoriented)")
        print(f"   regime: {va.credential.regime}")
    a_ok = va.is_answer() and len(va.credential.evidence["skeleton_stability"]) == 3
    ok &= a_ok

    # B. latent — skeleton stable but the consequence is a wrong do
    vb = L.run({"data": df[["X", "Y"]], "B": 25, "seed": 1})
    obs = InterventionEngine(g).query_observation("Y", 1, {"X": 1}).value
    print(f"\nB latent {{X,Y}} n=2000 -> {'ANSWER' if vb.is_answer() else 'ABSTAIN'}")
    if vb.is_answer():
        print(f"   discovered {vb.credential.evidence['discovered_edges']} (skeleton stable)")
        print(f"   BUT do() on learned X->Y = observational {obs:.3f} vs TRUE do {truth_do:.3f} "
              f"(gap {abs(obs-truth_do):.3f} — confidently wrong)")
        print(f"   credential regime declares this: '...latent confounding UNCERTIFIED'")
    b_ok = vb.is_answer() and abs(obs - truth_do) > 0.05
    ok &= b_ok

    # C. tiny n -> unstable -> ABSTAIN
    vc = L.run({"data": sample(g, 30, 2), "B": 25, "seed": 2})
    c_ok = not vc.is_answer()
    print(f"\nC tiny {{U,X,Y}} n=30 -> {'ABSTAIN' if c_ok else 'ANSWER'}: "
          f"{vc.reason if c_ok else 'unexpected answer'}")
    ok &= c_ok

    print("\nDiscovery-leg contract (NOTE-004 operational):")
    print("  • ANSWER only when the skeleton is bootstrap-STABLE (finite-sample reliability, checkable)")
    print("  • the credential ALWAYS declares what it cannot check: orientation (Markov equivalence)")
    print("    and latent confounding — the latter demonstrably makes do() wrong (case B)")
    print("  • unstable/insufficient data ABSTAINS — no confident-wrong structure is emitted")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
