"""Engine stress tests — answering gatekeeper self-audit 1.1 ('F-1 untested on
multi-confounder / small-sample / nonlinear') by ACTUALLY TESTING, not assuming.

Finding: the engine is an EXACT discrete-CPT enumerator. 'multi-confounder' is a
real generality test (passes). 'small-sample' and 'nonlinear' are benchmark-level
concerns (SCM generator), not engine-level — the engine does exact inference over
arbitrary CPT tables with no sampling. Conflating the two is where the audit had
water; these tests pin the engine-level truth."""
import itertools
import numpy as np
import pytest
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.causal.identify import find_adjustment_set


def _two_confounder_graph(seed=7):
    g = CausalGraph()
    for n in ("U1", "U2", "X", "Y"):
        g.add_variable(Variable(n))
    for e in (("U1", "X"), ("U2", "X"), ("U1", "Y"), ("U2", "Y"), ("X", "Y")):
        g.add_edge(*e)
    rng = np.random.default_rng(seed)
    g.set_cpt("U1", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("U2", {(): {1: 0.4, 0: 0.6}})
    for v in ("X", "Y"):
        order = list(g.parent_order(v))
        rows = {}
        for combo in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(0.1, 0.9)), 3)
            rows[combo] = {1: p, 0: 1 - p}
        g.set_cpt(v, rows)
    return g


class TestMultiConfounder:
    def test_two_confounders_adjustment_set_and_backdoor(self):
        """Engine generalizes to TWO confounders: auto-identifies {U1,U2} and the
        do-query equals the hand-computed backdoor formula exactly."""
        g = _two_confounder_graph()
        assert find_adjustment_set(g, "X", "Y") == {"U1", "U2"}
        eng = InterventionEngine(g)
        int_ate = (eng.query_intervention("Y", 1, {"X": 1}).value
                   - eng.query_intervention("Y", 1, {"X": 0}).value)
        # independent hand recomputation of sum_U [P(Y|X=1,U)-P(Y|X=0,U)] P(U)
        ordY = list(g.parent_order("Y"))
        manual = 0.0
        for u1, u2 in itertools.product((1, 0), repeat=2):
            pu = g.cpt("U1")[()][u1] * g.cpt("U2")[()][u2]
            k1 = tuple({"U1": u1, "U2": u2, "X": 1}[v] for v in ordY)
            k0 = tuple({"U1": u1, "U2": u2, "X": 0}[v] for v in ordY)
            manual += pu * (g.cpt("Y")[k1][1] - g.cpt("Y")[k0][1])
        assert int_ate == pytest.approx(manual, abs=1e-12)

    def test_confounding_actually_biases_observation(self):
        """Sanity: with multi-confounding, observational ATE != interventional ATE."""
        eng = InterventionEngine(_two_confounder_graph())
        obs = (eng.query_observation("Y", 1, {"X": 1}).value
               - eng.query_observation("Y", 1, {"X": 0}).value)
        intv = (eng.query_intervention("Y", 1, {"X": 1}).value
                - eng.query_intervention("Y", 1, {"X": 0}).value)
        assert abs(obs - intv) > 1e-6
