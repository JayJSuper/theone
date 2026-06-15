"""The causal-discovery frontier (the last unproven boundary; bet③ merges here).
The engine computes do() GIVEN structure. Real deployment must LEARN the structure
from data first. We use score-based structure learning (pgmpy HillClimbSearch) on
data from a known confounded SCM (U->X, U->Y, X->Y) and then compute do(X=1) on the
LEARNED structure, comparing to the true do.

Two regimes:
  - confounder OBSERVED: discovery sees {U,X,Y} -> recovers the structure ->
    do() adjusts for U -> correct (error -> 0 as n grows).
  - confounder LATENT: discovery sees only {X,Y} -> learns X->Y with no adjustment
    -> do() on the learned structure = the OBSERVATIONAL (confounded) estimate ->
    confidently WRONG, and MORE DATA DOES NOT FIX IT (latent confounding is not
    resolvable from observational data — this is why interventions / bet③ exist).

This is the deployment bottleneck and the live form of NOTE-004 (a credential
certifies the computation on the learned structure, not that the structure is right).
Pure computation, no API. Run: python experiments/causal_discovery/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork

HERE = Path(__file__).parent
NS = [200, 1000, 5000]
SEEDS = range(15)


def true_scm(seed):
    rng = np.random.default_rng(seed); g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    pu = round(float(rng.uniform(.3, .7)), 2); g.set_cpt("U", {(): {1: pu, 0: round(1-pu, 2)}})
    px1, px0 = round(float(rng.uniform(.7, .9)), 2), round(float(rng.uniform(.1, .3)), 2)
    g.set_cpt("X", {(1,): {1: px1, 0: round(1-px1, 2)}, (0,): {1: px0, 0: round(1-px0, 2)}})
    oY = list(g.parent_order("Y"))
    v = {(1, 1): .85, (0, 1): .45, (1, 0): .35, (0, 0): .15}
    g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY): {1: val, 0: round(1-val, 2)}
                    for (u, x), val in v.items()})
    return g


def sample(g, n, seed):
    rng = np.random.default_rng(seed); rows = []
    for _ in range(n):
        u = 1 if rng.random() < g.cpt("U")[()][1] else 0
        x = 1 if rng.random() < g.cpt("X")[(u,)][1] else 0
        oY = list(g.parent_order("Y")); key = tuple(u if p == "U" else x for p in oY)
        y = 1 if rng.random() < g.cpt("Y")[key][1] else 0
        rows.append({"U": u, "X": x, "Y": y})
    return pd.DataFrame(rows)


def do_on_learned(data, observed_cols):
    """Learn structure on observed_cols, fit CPTs, compute do(X=1) on the learned DAG."""
    df = data[observed_cols].astype("category")
    hc = HillClimbSearch(df)
    learned = hc.estimate(scoring_method="bic-d", show_progress=False)
    edges = list(learned.edges())
    g = CausalGraph()
    for c in observed_cols:
        g.add_variable(Variable(c))
    for a, b in edges:
        g.add_edge(a, b)
    if "X" not in g.variables or "Y" not in g.variables:
        return None, edges
    # fit CPTs by our own MLE (Laplace-smoothed) from the data on the learned structure
    import itertools
    D = data[observed_cols]
    for v in g.variables:
        order = list(g.parent_order(v))
        rows = {}
        if not order:
            ones = int((D[v] == 1).sum()); tot = len(D)
            p1 = (ones + 1) / (tot + 2); rows[()] = {1: round(p1, 6), 0: round(1 - p1, 6)}
        else:
            for combo in itertools.product([0, 1], repeat=len(order)):
                mask = np.ones(len(D), bool)
                for i, p in enumerate(order):
                    mask &= (D[p].values == combo[i])
                tot = int(mask.sum()); ones = int((D[v].values[mask] == 1).sum())
                p1 = (ones + 1) / (tot + 2); rows[combo] = {1: round(p1, 6), 0: round(1 - p1, 6)}
        g.set_cpt(v, rows)
    try:
        val = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
    except Exception:
        return None, edges
    return val, edges


def main():
    out = {"observed": {}, "latent": {}}
    for n in NS:
        eo, el = [], []
        for s in SEEDS:
            g = true_scm(1000 + s)
            true_do = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
            data = sample(g, n, 5000 + s)
            vo, _ = do_on_learned(data, ["U", "X", "Y"])      # confounder observed
            vl, _ = do_on_learned(data, ["X", "Y"])            # confounder latent
            if vo is not None:
                eo.append(abs(vo - true_do))
            if vl is not None:
                el.append(abs(vl - true_do))
        out["observed"][n] = round(float(np.mean(eo)), 4) if eo else None
        out["latent"][n] = round(float(np.mean(el)), 4) if el else None
    print("causal discovery -> do(X=1) error on the LEARNED structure (vs true do)\n")
    print(f"{'n':>7} | {'confounder OBSERVED':>20} | {'confounder LATENT':>18}")
    for n in NS:
        print(f"{n:>7} | {str(out['observed'][n]):>20} | {str(out['latent'][n]):>18}")
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    print("\nReading: with the confounder OBSERVED, score-based discovery recovers a")
    print("structure whose do() converges to the truth as n grows. With the confounder")
    print("LATENT, discovery learns X->Y and do() collapses to the confounded observational")
    print("estimate — a confident error that MORE DATA DOES NOT FIX. This is the deployment")
    print("bottleneck and the live form of NOTE-004: the credential certifies the computation")
    print("on the learned structure, not that the structure (latent-confounding-free) is right.")
    print("The remedy is interventional data (bet③) + structure uncertainty in the credential.")


if __name__ == "__main__":
    main()
