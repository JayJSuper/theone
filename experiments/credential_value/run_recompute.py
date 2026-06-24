"""任务七 A · 凭证可独立复算率: C(引擎) vs pgmpy(独立code path) 对 do(X=1)。
PREREG 冻结哈希: 71f176e4cd87101e228b1b7e3df408944c136b0be301c5c6f4f2ed35ea3dcb3d
Run: python experiments/credential_value/run_recompute.py
"""
from __future__ import annotations
import importlib.util, itertools, json, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
PREREG_SHA = "71f176e4cd87101e228b1b7e3df408944c136b0be301c5c6f4f2ed35ea3dcb3d"
assert hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() == PREREG_SHA, "PREREG changed!"
spec = importlib.util.spec_from_file_location("scale", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _a

import pytest  # noqa  (ensure env) -- but use importorskip-style
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def pgmpy_do(g, X, Y):
    """independent recompute of P(Y=1|do(X=1)): mutilate (cut X's parents) + clamp X=1 + VE."""
    edges = [(a, b) for a, b in g.nx.edges() if b != X]            # drop edges INTO X (graph surgery)
    m = DiscreteBayesianNetwork(edges)
    for v in g.variables:
        if v not in m.nodes():
            m.add_node(v)
    for v in g.variables:
        if v == X:
            m.add_cpds(TabularCPD(X, 2, [[0.0], [1.0]])); continue  # do(X=1)
        parents = list(g.parent_order(v))
        if not parents:
            p1 = g.cpt(v)[()][1]; m.add_cpds(TabularCPD(v, 2, [[1 - p1], [p1]]))
        else:
            cols = []
            for combo in itertools.product([0, 1], repeat=len(parents)):
                p1 = g.cpt(v)[tuple(combo)][1]; cols.append([1 - p1, p1])
            arr = list(map(list, zip(*cols)))
            m.add_cpds(TabularCPD(v, 2, arr, evidence=parents, evidence_card=[2] * len(parents)))
    m.check_model()
    return float(VariableElimination(m).query([Y], show_progress=False).values[1])


def main():
    print(f"PREREG {PREREG_SHA[:12]}…  task7-A credential recompute rate")
    rng = np.random.default_rng(20260614 + 12)
    match = 0; diffs = []; n = 150
    for i in range(n):
        d = scale.gen_dag(12, rng); g = scale.build_graph(d)
        c_val = float(scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value)
        p_val = pgmpy_do(g, d["x"], d["y"])
        diff = abs(c_val - p_val); diffs.append(diff)
        match += diff <= 1e-6
    rate = match / n
    out = {"n": n, "prereg_sha": PREREG_SHA,
           "credential_independent_recompute_rate": round(rate, 4),
           "max_abs_diff": float(np.max(diffs)), "mean_abs_diff": float(np.mean(diffs)),
           "tol": 1e-6, "verifier": "pgmpy VariableElimination (independent code path), graph-surgery do",
           "note": "每个 C 答案被独立实现(pgmpy)重算到 1e-6 容差内的占比 = 可复算凭证(LLM/统计基线不提供)"}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (HERE / "results_recompute.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sha = hashlib.sha256((HERE / "results_recompute.json").read_bytes()).hexdigest()
    print(f"\nresults_recompute.json sha256={sha[:16]}…")
    print(f">>> 凭证可独立复算率 = {rate:.4f} ({match}/{n}, 容差1e-6)  max差={np.max(diffs):.2e}")


if __name__ == "__main__":
    main()
