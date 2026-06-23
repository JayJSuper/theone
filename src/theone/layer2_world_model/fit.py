"""Fit CPTs on a (discovered or expert-oriented) structure — the bridge from a learned
graph to a do()-computable CausalGraph.

Honest note on orientation: structure discovery from observational data returns a
SKELETON (the orientation is only identified up to the Markov equivalence class). To
compute do() we need a DAG. This module accepts an explicit `oriented_edges` (expert /
interventional knowledge — a source beyond observational data) and fits Laplace-smoothed
CPTs by MLE. The resulting graph carries the do()-engine's usual 'structure-assumed'
limit, now with orientation explicitly sourced rather than silently guessed.
"""
from __future__ import annotations
import itertools
import pandas as pd

from theone.types import Variable
from theone.causal.graph import CausalGraph


def fit_cpts(df: pd.DataFrame, oriented_edges, smoothing: float = 1.0) -> CausalGraph:
    nodes = list(df.columns)
    g = CausalGraph()
    for n in nodes:
        g.add_variable(Variable(n))
    for a, b in oriented_edges:
        g.add_edge(a, b)
    for v in nodes:
        parents = list(g.parent_order(v))
        cpt = {}
        if not parents:
            n1 = float((df[v] == 1).sum()) + smoothing
            n0 = float((df[v] == 0).sum()) + smoothing
            cpt[()] = {0: n0 / (n0 + n1), 1: n1 / (n0 + n1)}
        else:
            for combo in itertools.product([0, 1], repeat=len(parents)):
                mask = (df[parents] == list(combo)).all(axis=1)
                sub = df[mask][v]
                n1 = float((sub == 1).sum()) + smoothing
                n0 = float((sub == 0).sum()) + smoothing
                cpt[tuple(combo)] = {0: n0 / (n0 + n1), 1: n1 / (n0 + n1)}
        g.set_cpt(v, cpt)
    g.validate()
    return g


__all__ = ["fit_cpts"]
