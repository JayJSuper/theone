"""F-1 frozen regression: three-node confounded graph U->X, U->Y, X->Y.
Truth table source: gatekeeper R3 review (2026-06-11), seven assertions frozen.
Per T1_WORKORDER_F1_FIX.md - this file is the MVP-2A sanity gate (CI: required).
"""
import pytest
from theone.types import Variable, TheOneConfig
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

TOL = 1e-6

# ---- FROZEN parameters (registry: T1; do not modify) ----
FROZEN = dict(
    p_u1=0.5,
    p_x1_u={1: 0.8, 0: 0.2},
    p_y1_xu={(1, 1): 0.9, (1, 0): 0.5, (0, 1): 0.6, (0, 0): 0.2},
)

# ---- A8 mechanism-guard perturbed parameters (expected values computed live) ----
PERTURBED = dict(
    p_u1=0.3,
    p_x1_u={1: 0.7, 0: 0.1},
    p_y1_xu={(1, 1): 0.85, (1, 0): 0.4, (0, 1): 0.55, (0, 0): 0.15},
)


def analytic_truth(p):
    """Analytic truth: observation uses POSTERIOR P(U|X), intervention uses PRIOR
    P(U). Direct transcription of the gatekeeper's R1 formulas - shares zero code
    with the implementation under test."""
    out = {}
    for x in (1, 0):
        px_u1 = p["p_x1_u"][1] if x == 1 else 1 - p["p_x1_u"][1]
        px_u0 = p["p_x1_u"][0] if x == 1 else 1 - p["p_x1_u"][0]
        p_x = px_u1 * p["p_u1"] + px_u0 * (1 - p["p_u1"])
        post_u1 = px_u1 * p["p_u1"] / p_x
        y11, y10 = p["p_y1_xu"][(x, 1)], p["p_y1_xu"][(x, 0)]
        out[f"obs_{x}"] = y11 * post_u1 + y10 * (1 - post_u1)
        out[f"do_{x}"] = y11 * p["p_u1"] + y10 * (1 - p["p_u1"])
    out["obs_ate"] = out["obs_1"] - out["obs_0"]
    out["do_ate"] = out["do_1"] - out["do_0"]
    return out


def build_engine(p) -> InterventionEngine:
    g = CausalGraph()
    for name in ("U", "X", "Y"):
        g.add_variable(Variable(name))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: p["p_u1"], 0: 1 - p["p_u1"]}})
    g.set_cpt("X", {(1,): {1: p["p_x1_u"][1], 0: 1 - p["p_x1_u"][1]},
                    (0,): {1: p["p_x1_u"][0], 0: 1 - p["p_x1_u"][0]}})
    # parent_order("Y") = ("U", "X") sorted -> key (u, x)
    g.set_cpt("Y", {(u, x): {1: p["p_y1_xu"][(x, u)], 0: 1 - p["p_y1_xu"][(x, u)]}
                    for u in (0, 1) for x in (0, 1)})
    return InterventionEngine(g)


class TestF1ConfoundingRegression:
    def test_frozen_truth_table(self):
        """A1-A7: frozen truth table 0.82/0.70/0.28/0.40/0.54/0.30/True."""
        eng = build_engine(FROZEN)
        assert eng.query_observation("Y", 1, {"X": 1}).value == pytest.approx(0.82, abs=TOL)  # A1
        assert eng.query_intervention("Y", 1, {"X": 1}).value == pytest.approx(0.70, abs=TOL)  # A2
        assert eng.query_observation("Y", 1, {"X": 0}).value == pytest.approx(0.28, abs=TOL)  # A3
        assert eng.query_intervention("Y", 1, {"X": 0}).value == pytest.approx(0.40, abs=TOL)  # A4
        assert eng.observational_ate("X", "Y") == pytest.approx(0.54, abs=TOL)                 # A5
        assert eng.interventional_ate("X", "Y") == pytest.approx(0.30, abs=TOL)                # A6
        assert eng.compare("X", "Y", TheOneConfig()).are_different is True                     # A7

    def test_frozen_matches_analytic(self):
        """Guard of the guard: analytic function must reproduce frozen values."""
        t = analytic_truth(FROZEN)
        for key, val in [("obs_1", .82), ("do_1", .70), ("obs_0", .28),
                         ("do_0", .40), ("obs_ate", .54), ("do_ate", .30)]:
            assert t[key] == pytest.approx(val, abs=TOL)

    def test_mechanism_guard_perturbed(self):
        """A8: perturbed parameter set - locks the MECHANISM, not the numbers;
        hard-coding the frozen table cannot pass this."""
        t = analytic_truth(PERTURBED)
        eng = build_engine(PERTURBED)
        assert eng.query_observation("Y", 1, {"X": 1}).value == pytest.approx(t["obs_1"], abs=TOL)
        assert eng.query_intervention("Y", 1, {"X": 1}).value == pytest.approx(t["do_1"], abs=TOL)
        assert eng.query_observation("Y", 1, {"X": 0}).value == pytest.approx(t["obs_0"], abs=TOL)
        assert eng.query_intervention("Y", 1, {"X": 0}).value == pytest.approx(t["do_0"], abs=TOL)
        assert eng.compare("X", "Y").are_different is True
