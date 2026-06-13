"""Decisive reproducible case for REPORT.md §2: missing a STRONG confounder yields
a large SILENT bias (0.30), with a valid 'computation-exact' credential. Shows the
wrong-structure bias is bounded only by confounder strength — no upper limit, no
warning. The credential certifies computation, not structure.

Run: python experiments/wrong_structure/strong_confounder_demo.py
"""
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine


def strong_scm():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("X", {(1,): {1: 0.9, 0: 0.1}, (0,): {1: 0.1, 0: 0.9}})   # U strongly drives X
    oY = list(g.parent_order("Y"))
    vals = {(1, 1): 0.95, (1, 0): 0.90, (0, 1): 0.20, (0, 0): 0.10}    # (u,x): U strongly drives Y
    g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY): {1: v, 0: 1 - v}
                    for (u, x), v in vals.items()})
    return g


def main():
    eng = InterventionEngine(strong_scm())
    true_do = eng.query_intervention("Y", 1, {"X": 1}).value
    missing = eng.query_observation("Y", 1, {"X": 1}).value
    print(f"true do(X=1)             = {true_do:.4f}")
    print(f"missing-confounder est   = {missing:.4f}  (observational P(Y|X=1))")
    print(f"SILENT bias              = {abs(missing - true_do):.4f}")
    print("Credential certifies computation is exact — NOT that the structure is correct.")
    assert abs(abs(missing - true_do) - 0.30) < 1e-6


if __name__ == "__main__":
    main()
