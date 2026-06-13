"""AM-017 backfill: independent pgmpy cross-check of the SCALE AXIS (second vital
number) — all 900 stored engine truths (450 fixed-skeleton + 450 random-DAG),
re-derived from the frozen seeds and recomputed by pgmpy via independent surgery.

General binary-DAG translator (up to 12 nodes, <=3 parents). Completes the
three-vital-number oracle requirement (cliff 300/300, F-1 7/7, scale 900/900).

Run: python experiments/oracle_crosscheck/scale_oracle.py
"""
from __future__ import annotations
import importlib.util, itertools, json
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

ROOT = Path(__file__).parent.parent.parent
_s = importlib.util.spec_from_file_location("scalerun", ROOT / "experiments" / "baseline_b_v1_scale" / "run.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def to_pgmpy(g):
    nodes = list(g.variables)
    edges = [(p, v) for v in nodes for p in g.parent_order(v)]
    m = DiscreteBayesianNetwork(edges)
    for v in nodes:
        if v not in m.nodes():
            m.add_node(v)
    cpds = []
    for v in nodes:
        order = list(g.parent_order(v)); sn = {v: [0, 1]}
        for p in order:
            sn[p] = [0, 1]
        if not order:
            p1 = g.cpt(v)[()][1]
            cpds.append(TabularCPD(v, 2, [[1 - p1], [p1]], state_names={v: [0, 1]}))
        else:
            combos = list(itertools.product([0, 1], repeat=len(order)))
            r1 = [g.cpt(v)[c][1] for c in combos]; r0 = [g.cpt(v)[c][0] for c in combos]
            cpds.append(TabularCPD(v, 2, [r0, r1], evidence=order,
                                   evidence_card=[2] * len(order), state_names=sn))
    m.add_cpds(*cpds); assert m.check_model()
    return m


def pgmpy_do1(g, x, y):
    """do(x=1): independent surgery — drop edges into x, pin x=1, query y."""
    m = to_pgmpy(g)
    for p in list(g.parent_order(x)):
        m.remove_edge(p, x)
    m.remove_cpds(m.get_cpds(x))
    m.add_cpds(TabularCPD(x, 2, [[0.0], [1.0]], state_names={x: [0, 1]}))
    assert m.check_model()
    return float(VariableElimination(m).query([y], show_progress=False).values[1])


def main():
    modes = [("fixed", S.gen_fixed_skeleton, 20260615, "rows.fixed.jsonl"),
             ("random", S.gen_dag, 20260614, "rows.random.jsonl")]
    overall_max = 0.0; total = 0; report = {}
    for mode, gen, seed, rowfile in modes:
        stored = [json.loads(l) for l in (ROOT / "experiments" / "baseline_b_v1_scale" / rowfile).read_text().splitlines()]
        by = {(r["tier"], r["i"]): r["truth"] for r in stored}
        mode_max = 0.0; n = 0; eng_vs_stored_max = 0.0
        for tier, n_nodes in S.SIZES.items():
            rng = np.random.default_rng(seed + n_nodes)
            for i in range(150):   # n_per_tier in the frozen formal runs
                d = gen(n_nodes, rng)
                g = S.build_graph(d)
                eng = round(InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value, 6)
                pg = pgmpy_do1(g, d["x"], d["y"])
                mode_max = max(mode_max, abs(pg - eng))
                if (tier, i) in by:
                    eng_vs_stored_max = max(eng_vs_stored_max, abs(eng - by[(tier, i)]))
                n += 1
        overall_max = max(overall_max, mode_max); total += n
        report[mode] = {"n": n, "max_pgmpy_engine_diff": round(mode_max, 10),
                        "max_engine_vs_stored_diff": round(eng_vs_stored_max, 10)}
        print(f"{mode:7}: {n} SCMs | max|pgmpy-engine|={mode_max:.2e} | max|engine-stored|={eng_vs_stored_max:.2e}")
    verdict = ("PASS — scale-axis engine truths independently confirmed by pgmpy <1e-6"
               if overall_max < 1e-6 else "FAIL")
    print(f"\nTOTAL {total} SCMs · max|pgmpy-engine| = {overall_max:.2e} -> {verdict}")
    (Path(__file__).parent / "scale_cross_validation.json").write_text(json.dumps(
        {"total_scms": total, "max_abs_diff": overall_max, "per_mode": report,
         "oracle": f"pgmpy {__import__('pgmpy').__version__}", "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
