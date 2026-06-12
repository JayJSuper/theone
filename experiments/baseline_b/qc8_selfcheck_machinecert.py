"""R3 v3 machine-cert of Jack's Q-C8 self-check (Baseline B v1 design).

Two parts:
(A) Qualitative causal identification (exercise facilities / BMI / land value)
    — certified against the project's own identify module (not by eye):
    1-3: confounder structure, backdoor path, adjustment set = {land_value}
    4:   land_value unobserved -> backdoor blocked -> must refuse (gate).
(B) Power-analysis arithmetic (the only numeric claims): n ~= 131 -> +10% -> 145
    -> rounded 150 per cell.
Exit 0 iff all certified.
"""
from __future__ import annotations
import math
import sys
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.identify import backdoor_paths, find_adjustment_set
from theone.hybrid import LibraryGraph, identify_gate, _confounded

ok = True
def check(name, cond):
    global ok
    print(f"{'PASS' if cond else 'FAIL'}  {name}")
    ok = ok and cond

# ---- (A) identification, certified by the engine --------------------------
g = CausalGraph()
for n in ("land_value", "facilities", "bmi"):
    g.add_variable(Variable(n))
g.add_edge("land_value", "facilities")   # U -> X
g.add_edge("land_value", "bmi")          # U -> Y
g.add_edge("facilities", "bmi")          # X -> Y
g.set_cpt("land_value", {(): {1: 0.5, 0: 0.5}})
g.set_cpt("facilities", {(1,): {1: 0.7, 0: 0.3}, (0,): {1: 0.3, 0: 0.7}})
order = list(g.parent_order("bmi"))
vals = {(1, 1): 0.3, (0, 1): 0.5, (1, 0): 0.5, (0, 0): 0.7}
g.set_cpt("bmi", {tuple(u if p == "land_value" else x for p in order): {1: v, 0: 1 - v}
                  for (u, x), v in vals.items()})

paths = backdoor_paths(g, "facilities", "bmi")
check("Q2: backdoor path is X<-U->Y",
      paths == [["facilities", "land_value", "bmi"]])
check("Q3: adjustment set == {land_value}",
      find_adjustment_set(g, "facilities", "bmi") == {"land_value"})

lg = LibraryGraph(key="bmi_case", treatment="facilities", outcome="bmi",
                  confounder="land_value", aliases={}, evidence_tier="machine_validated",
                  note="QC8 self-check", latent=("land_value",), graph=g)
gate = identify_gate(lg)
check("Q4: U unobserved -> NOT identifiable via backdoor (must refuse)",
      gate["identifiable"] is False and gate["missing"] == ["land_value"])

# ---- (B) power-analysis arithmetic ----------------------------------------
z_a, z_b = 1.6449, 0.8416                  # alpha=.05 one-sided, power=.80
p1, p2 = 0.5, 0.65
n = (z_a + z_b) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
check(f"n base ~= 131 (got {n:.1f})", abs(n - 131) < 1.5)
n_drop = math.ceil(n) * 1.10
check(f"+10% dropout ~= 145 (got {n_drop:.1f})", abs(n_drop - 145) < 1.5)
check("rounded cell size 150 >= required", 150 >= n_drop)

print("\nVERDICT:", "PASS — Q-C8 self-check machine-certified" if ok else "FAIL")
sys.exit(0 if ok else 1)
