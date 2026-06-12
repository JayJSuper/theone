"""Front-door estimator tests (Q-C14). The crown test: the front-door estimate,
using ONLY observed {X,M,Y}, must equal the TRUE do-effect that graph surgery
computes using the unobserved confounder U — i.e. it recovers the truth without
ever observing U."""
import itertools
import numpy as np
import pytest
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.causal.estimators import frontdoor_prob, frontdoor_ate


def smoking_tar_cancer(seed=3):
    """U -> X(smoking) -> M(tar) -> Y(cancer), U -> Y. Front-door identifiable;
    U is the (would-be unobserved) confounder. Random CPTs."""
    rng = np.random.default_rng(seed)
    g = CausalGraph()
    for n in ("U", "X", "M", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("X", "M"); g.add_edge("M", "Y")
    g.add_edge("U", "Y")
    g.set_cpt("U", {(): {1: 0.4, 0: 0.6}})
    for v, parents in (("X", ["U"]), ("M", ["X"])):
        order = list(g.parent_order(v))
        rows = {}
        for combo in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(0.1, 0.9)), 3)
            rows[combo] = {1: p, 0: 1 - p}
        g.set_cpt(v, rows)
    order = list(g.parent_order("Y"))           # parents M, U
    rows = {}
    for combo in itertools.product((1, 0), repeat=len(order)):
        p = round(float(rng.uniform(0.1, 0.9)), 3)
        rows[combo] = {1: p, 0: 1 - p}
    g.set_cpt("Y", rows)
    return g


class TestFrontDoorRecoversTruth:
    @pytest.mark.parametrize("seed", [3, 11, 42, 99])
    def test_frontdoor_equals_graph_surgery_truth(self, seed):
        eng = InterventionEngine(smoking_tar_cancer(seed))
        # ground truth via graph surgery (uses U)
        true1 = eng.query_intervention("Y", 1, {"X": 1}).value
        true0 = eng.query_intervention("Y", 1, {"X": 0}).value
        # front-door estimate uses ONLY X, M, Y observational quantities
        est1 = frontdoor_prob(eng, "X", "Y", "M", 1, 1)
        est0 = frontdoor_prob(eng, "X", "Y", "M", 1, 0)
        assert est1 == pytest.approx(true1, abs=1e-9)
        assert est0 == pytest.approx(true0, abs=1e-9)
        assert frontdoor_ate(eng, "X", "Y", "M") == pytest.approx(true1 - true0, abs=1e-9)

    def test_frontdoor_differs_from_naive_observation(self):
        """Sanity: the front-door ATE != naive observational ATE (confounding real)."""
        eng = InterventionEngine(smoking_tar_cancer(3))
        obs = (eng.query_observation("Y", 1, {"X": 1}).value
               - eng.query_observation("Y", 1, {"X": 0}).value)
        assert abs(frontdoor_ate(eng, "X", "Y", "M") - obs) > 1e-6
