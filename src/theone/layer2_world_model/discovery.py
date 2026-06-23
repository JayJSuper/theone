"""Causal structure discovery + bootstrap stability — the L2 discovery leg.

The engine computes do() GIVEN a structure; deployment must LEARN the structure first,
and that is the unverified frontier (NOTE-004). This module discovers a candidate DAG
(pgmpy HillClimbSearch / BIC) and measures how STABLE each edge is under bootstrap
resampling — a truth-free reliability signal (analogous to probe-5's subset-spread).

The honest boundary, made explicit: bootstrap stability catches FINITE-SAMPLE noise
(checkable) but NOT latent confounding (a latent confounder yields a stably WRONG
structure from observational data — undetectable here). So discovery is 'sample-stable,
latent-confounding-uncertified': the layer gates instability it can see and declares
the limit it cannot.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

# pgmpy 1.1.x deprecation FutureWarnings are issued with stacklevel pointing at the
# caller, so we filter by the specific messages (works whether this module is imported
# directly or via the package). We keep the estimators-path HillClimbSearch because it
# exposes the .estimate(scoring_method=...) API we rely on (the causal_discovery
# relocation is a different, incompatible class).
for _msg in (r".*HillClimbSearch is deprecated.*", r".*StructureScore.*deprecated.*"):
    warnings.filterwarnings("ignore", message=_msg, category=FutureWarning)
from pgmpy.estimators import HillClimbSearch


def discover(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Score-based structure learning (deterministic given the data). Columns are
    coerced to categorical so pgmpy uses the discrete BIC score."""
    disc = df.astype("category")
    hc = HillClimbSearch(disc)
    dag = hc.estimate(scoring_method="bic-d", show_progress=False)
    return sorted((str(a), str(b)) for a, b in dag.edges())


def _skeleton(edges) -> set[frozenset]:
    return {frozenset((a, b)) for a, b in edges}


def bootstrap_stability(df: pd.DataFrame, B: int = 25, seed: int = 0) -> dict:
    """Resample rows with replacement B times, re-discover, and report how often each
    directed edge and each undirected skeleton link reappears (frequency in [0,1])."""
    rng = np.random.default_rng(seed)
    n = len(df)
    point_skel = _skeleton(discover(df))          # full-data skeleton (the point estimate)
    edge_counts: dict[tuple, int] = {}
    skel_counts: dict[frozenset, int] = {}
    agree = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        e = discover(df.iloc[idx].reset_index(drop=True))
        bs = _skeleton(e)
        if bs == point_skel:                       # exact skeleton reproduced?
            agree += 1
        for ed in e:
            edge_counts[ed] = edge_counts.get(ed, 0) + 1
        for sk in bs:
            skel_counts[sk] = skel_counts.get(sk, 0) + 1
    return {
        "edge_freq": {ed: c / B for ed, c in edge_counts.items()},
        "skeleton_freq": {tuple(sorted(sk)): c / B for sk, c in skel_counts.items()},
        "skeleton_agreement": agree / B,           # fraction of resamples = point skeleton
        "B": B,
    }


__all__ = ["discover", "bootstrap_stability"]
