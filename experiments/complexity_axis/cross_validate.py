"""Independent third-party cross-validation of the cliff experiment's engine
truths. Reproduces all 300 SCMs deterministically (same seeds as run.py), rebuilds
each as a pgmpy DiscreteBayesianNetwork with IDENTICAL CPTs, computes
P(Y=1|do(X=1)) via pgmpy's CausalInference, and compares to our engine's `truth`.

Purpose: convert "engine = 1.000" from "trust our engine" to "verified against an
independent causal-inference library to 1e-6". Defends against the skeptic attack
"how do we know your engine's truth isn't just self-consistent?"

Self-validates the CPT translation first (pgmpy must reproduce our engine's
OBSERVATIONAL conditionals) so an ordering bug can't pass silently.

Run: source ~/.theone_keys.env(not needed) && python .../cross_validate.py
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

# import the exact generator from run.py
import importlib.util
HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("cliffrun", HERE / "run.py")
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import CausalInference


def to_pgmpy(g, k):
    Us = [f"U{i}" for i in range(k)]
    edges = [(u, "X") for u in Us] + [(u, "Y") for u in Us] + [("X", "Y")]
    # W is isolated; add as a node with no edges
    model = DiscreteBayesianNetwork(edges)
    model.add_node("W")
    cpds = []
    for v in g.variables:
        order = list(g.parent_order(v))
        sn = {v: [0, 1]}
        for p in order:
            sn[p] = [0, 1]
        if not order:
            p1 = g.cpt(v)[()][1]
            cpds.append(TabularCPD(v, 2, [[1 - p1], [p1]], state_names={v: [0, 1]}))
        else:
            # columns enumerated with evidence[0] slowest -> itertools.product([0,1],repeat=n)
            combos = list(itertools.product([0, 1], repeat=len(order)))
            row1 = [g.cpt(v)[c][1] for c in combos]
            row0 = [g.cpt(v)[c][0] for c in combos]
            cpds.append(TabularCPD(v, 2, [row0, row1], evidence=order,
                                   evidence_card=[2] * len(order), state_names=sn))
    model.add_cpds(*cpds)
    assert model.check_model()
    return model


def main():
    # ---- self-validate CPT translation on a few SCMs (observational conditionals) ----
    print("[self-validate] pgmpy vs engine on observational conditionals...")
    from pgmpy.inference import VariableElimination
    for (k, i) in [(1, 0), (2, 3), (3, 7), (4, 1), (5, 2)]:
        g = R.k_graph(k, R.BASE_SEED + 1000 * k + i)
        model = to_pgmpy(g, k)
        ve = VariableElimination(model)
        eng = InterventionEngine(g)
        # P(Y=1|X=1) observational
        p_pg = ve.query(["Y"], evidence={"X": 1}, show_progress=False).values[1]
        p_en = eng.query_observation("Y", 1, {"X": 1}).value
        assert abs(p_pg - p_en) < 1e-6, f"obs mismatch k{k}i{i}: pgmpy {p_pg} vs engine {p_en}"
    print("  OK — CPT translation faithful (observational conditionals match 1e-6)\n")

    # ---- cross-validate every do-query against stored engine truth ----
    rows = [json.loads(l) for l in (HERE / "rows.jsonl").read_text().splitlines() if l.strip()]
    by_ki = {(r["k"], r["i"]): r for r in rows}
    max_diff = 0.0; worst = None; n = 0
    per_k_max = {}
    for k in R.KS:
        for i in range(R.N_PER_K):
            g = R.k_graph(k, R.BASE_SEED + 1000 * k + i)
            model = to_pgmpy(g, k)
            ci = CausalInference(model)
            q = ci.query(variables=["Y"], do={"X": 1}, show_progress=False)
            p_pgmpy = float(q.values[1])
            truth = by_ki[(k, i)]["truth"]
            d = abs(p_pgmpy - truth)
            per_k_max[k] = max(per_k_max.get(k, 0.0), d)
            if d > max_diff:
                max_diff = d; worst = (k, i, p_pgmpy, truth)
            n += 1
        print(f"  k={k} (2^{k}={2**k}): {R.N_PER_K} SCMs cross-checked, max|pgmpy-engine|={per_k_max[k]:.2e}")
    print(f"\n===== CROSS-VALIDATION ({n} SCMs) =====")
    print(f"max |pgmpy do(X=1) - engine truth| = {max_diff:.2e}")
    print(f"worst case: k={worst[0]} i={worst[1]} pgmpy={worst[2]:.6f} engine={worst[3]:.6f}")
    verdict = "PASS — engine truths independently confirmed by pgmpy to <1e-6" if max_diff < 1e-6 \
        else "FAIL — discrepancy exceeds 1e-6, investigate"
    print(verdict)
    (HERE / "cross_validation.json").write_text(json.dumps(
        {"n_scms": n, "max_abs_diff": max_diff, "per_k_max": per_k_max,
         "worst": {"k": worst[0], "i": worst[1], "pgmpy": worst[2], "engine": worst[3]},
         "oracle": f"pgmpy {__import__('pgmpy').__version__}", "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
