"""Fusion integration v2 · the data-to-do pipeline through one spine.

Chains the new L2 legs: CausalDiscoveryLayer (data -> stable skeleton) ->
StructureFitLayer (expert orientation must match the discovered skeleton; fit CPTs) ->
CausalLayer (do() + pgmpy IPRG + E-value). A system ANSWER is a stacked receipt:
discovery stability + fit reproducibility + do recomputation + sensitivity bound.

  A. healthy data + correct orientation -> SYSTEM ANSWER (3 stacked credentials).
  B. insufficient data (n=30)           -> ABSTAIN at discovery (unstable skeleton).
  C. orientation contradicts the data    -> ABSTAIN at structure-fit.

Run:  .venv/bin/python experiments/fusion_integration_v2/run.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.core.spine import Spine
from theone.layer2_world_model import CausalDiscoveryLayer, StructureFitLayer, CausalLayer

PY = {(1, 1): .85, (0, 1): .45, (1, 0): .35, (0, 0): .15}
TRUE_EDGES = [("U", "X"), ("U", "Y"), ("X", "Y")]


def true_graph():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    for a, b in TRUE_EDGES:
        g.add_edge(a, b)
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
    y = np.array([1 if rng.random() < PY[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    return pd.DataFrame({"U": u, "X": x, "Y": y})


def run(label, ctx):
    layers = [CausalDiscoveryLayer(), StructureFitLayer(), CausalLayer()]
    sv = Spine(layers).run(ctx)
    if sv.is_answer():
        do_cred = sv.credentials[-1]
        s = do_cred.evidence["sensitivity"]
        print(f"  {label:<26} SYSTEM ANSWER | {len(sv.credentials)} creds: "
              f"{' -> '.join(c.layer for c in sv.credentials)}")
        print(f"  {'':<26} do(X=1)={do_cred.value:.3f}  E-value={s['e_value']}")
    else:
        print(f"  {label:<26} SYSTEM ABSTAIN @ {sv.abstained_at}: {sv.reason}")
    return sv


def main():
    print("=== Fusion integration v2: data -> discover -> fit -> do (one spine) ===\n")
    g = true_graph()
    truth = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
    print(f"(reference) true do(X=1) = {truth:.3f}\n")

    base = {"oriented_edges": TRUE_EDGES, "treatment": "X", "target": "Y", "B": 25, "seed": 1}

    a = run("A healthy n=2000", {**base, "data": sample(g, 2000, 0)})
    b = run("B insufficient n=30", {**base, "data": sample(g, 30, 2)})
    # C: orientation DROPS the U->Y link, so its skeleton no longer matches the data
    c = run("C orientation != data", {**base, "oriented_edges": [("U", "X"), ("X", "Y")],
                                      "data": sample(g, 2000, 0)})

    ok = (a.is_answer() and abs(a.credentials[-1].value - truth) < 0.05
          and not b.is_answer()                       # insufficient data -> refused (any layer)
          and not c.is_answer() and c.abstained_at == "L2f_structure_fit")
    print("\nIntegration v2 contract: a data-to-do answer stacks discovery-stability + fit-")
    print("reproducibility + do-recomputation + E-value; insufficient data or data-inconsistent")
    print("orientation trips the abstain bus before any do() is emitted.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
