"""Test suite for graph, identify, SCM generator, EG/A7, memory, agent, CLI.
Frozen fixed cases come from gatekeeper-signed self-checks (registry: Q-C5, Q-C6,
delta_min scheme) - machine-verified before freezing (R3 v3)."""
import json
import numpy as np
import pytest
from theone.types import (Variable, TheOneConfig, GraphValidationError,
                          CalibrationRequiredError)
from theone.causal.graph import CausalGraph
from theone.causal.identify import (backdoor_paths, check_backdoor,
                                    find_adjustment_set)
from theone.causal.engine import InterventionEngine
from theone.bench.eg import (SCMSpec, SCMGenerator, abs_errors, eg_score,
                             conjunctive_verdict, a7_from_summary, a7_judgment)
from theone.bench.runner import (run_calibration, run_frozen,
                                 _backdoor_adjusted_int_estimate,
                                 _assoc_int_estimate)
from theone.memory.store import MemoryStore
from theone.agent.orchestrator import Orchestrator
from theone.cli import main as cli_main


# ---------- helpers ----------
def confounded_graph():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("X", {(1,): {1: 0.8, 0: 0.2}, (0,): {1: 0.2, 0: 0.8}})
    g.set_cpt("Y", {(1, 1): {1: 0.9, 0: 0.1}, (0, 1): {1: 0.5, 0: 0.5},
                    (1, 0): {1: 0.6, 0: 0.4}, (0, 0): {1: 0.2, 0: 0.8}})
    return g


# ---------- CC-03 graph ----------
class TestGraph:
    def test_cycle_rejected(self):
        g = CausalGraph()
        g.add_variable(Variable("A")); g.add_variable(Variable("B"))
        g.add_edge("A", "B")
        with pytest.raises(GraphValidationError):
            g.add_edge("B", "A")

    def test_out_of_range_probability_rejected_not_rescaled(self):
        """Frozen ruling: validation replaces normalization - 1e6 must be REJECTED."""
        g = CausalGraph(); g.add_variable(Variable("A"))
        with pytest.raises(GraphValidationError):
            g.set_cpt("A", {(): {1: 1e6, 0: -999999.0}})
        with pytest.raises(GraphValidationError):
            g.set_cpt("A", {(): {1: 1e-6, 0: 1e-6}})  # doesn't sum to 1

    def test_validate_missing_cpt(self):
        g = CausalGraph(); g.add_variable(Variable("A"))
        with pytest.raises(GraphValidationError):
            g.validate()

    def test_content_hash_stable_and_sensitive(self):
        g1, g2 = confounded_graph(), confounded_graph()
        assert g1.content_hash() == g2.content_hash()
        g2.set_cpt("U", {(): {1: 0.4, 0: 0.6}})
        assert g1.content_hash() != g2.content_hash()


# ---------- CC-04 identify ----------
class TestIdentify:
    def test_confounded_graph_adjustment_is_U(self):
        g = confounded_graph()
        assert backdoor_paths(g, "X", "Y") == [["X", "U", "Y"]]
        assert find_adjustment_set(g, "X", "Y") == {"U"}

    def test_chain_no_backdoor(self):
        g = CausalGraph()
        for n in ("A", "B", "C"):
            g.add_variable(Variable(n))
        g.add_edge("A", "B"); g.add_edge("B", "C")
        assert backdoor_paths(g, "A", "C") == []
        assert find_adjustment_set(g, "A", "C") == set()

    def test_collider_descendant_violates_backdoor(self):
        g = CausalGraph()
        for n in ("X", "Y", "C"):
            g.add_variable(Variable(n))
        g.add_edge("X", "Y"); g.add_edge("X", "C"); g.add_edge("Y", "C")
        assert find_adjustment_set(g, "X", "Y") == set()
        assert check_backdoor(g, "X", "Y", {"C"}) is False  # descendant of X

    def test_m_graph_empty_set_valid_conditioning_M_opens(self):
        g = CausalGraph()
        for n in ("U1", "U2", "M", "X", "Y"):
            g.add_variable(Variable(n))
        g.add_edge("U1", "X"); g.add_edge("U1", "M")
        g.add_edge("U2", "M"); g.add_edge("U2", "Y"); g.add_edge("X", "Y")
        assert check_backdoor(g, "X", "Y", set()) is True       # collider M blocks
        assert check_backdoor(g, "X", "Y", {"M"}) is False      # conditioning opens
        assert find_adjustment_set(g, "X", "Y") == set()


# ---------- CC-07 SCM generator ----------
class TestSCMGenerator:
    def test_qc5_selfcheck_bias_equals_coefficient_product(self):
        """Frozen Q-C5 self-check: beta 0.6 x 0.5 -> obs slope 0.6, int 0.3, bias 0.3."""
        spec = SCMSpec("linear_gaussian_3node", 0.6, 0.5, 0.3, 0.5, 200_000, 7)
        scm = SCMGenerator().generate(spec)
        assert scm.true_int_ate() == pytest.approx(0.30)
        assert scm.true_obs_slope() == pytest.approx(0.60)
        assert scm.true_bias() == pytest.approx(0.30)
        d = scm.sample()
        ols = float(np.cov(d["X"], d["Y"], bias=True)[0, 1] / np.var(d["X"]))
        assert ols == pytest.approx(0.60, abs=0.02)  # empirical matches analytic

    def test_seed_reproducibility_and_fingerprint(self):
        spec = SCMSpec("linear_gaussian_3node", 0.4, 0.4, 0.2, 0.5, 1000, 11)
        a = SCMGenerator().generate(spec).sample()
        b = SCMGenerator().generate(spec).sample()
        assert np.allclose(a["Y"], b["Y"])
        assert len(spec.fingerprint()) == 16

    def test_grid_injected_not_hardcoded(self):
        cfg = {"beta_ux": [0.2], "beta_uy": [0.3], "beta_xy": [0.1],
               "noise": [0.5], "n_samples": 100, "base_seed": 1}
        specs = list(SCMGenerator().grid(cfg))
        assert len(specs) == 1 and specs[0].beta_ux == 0.2


# ---------- CC-08 EG / A7 ----------
class TestEGAndA7:
    def test_abs_errors_and_eg(self):
        e = abs_errors([0.1, 0.2], [0.0, 0.0])
        assert e["mae"] == pytest.approx(0.15)
        assert eg_score(0.1, 0.3) == pytest.approx(3.0)
        assert eg_score(0.1, 0.3, near_zero=True) is None  # Amendment 1 near-zero

    def test_conjunctive_verdict_amendment_1a(self):
        assert conjunctive_verdict(True, True, True, True) == "effective"
        assert conjunctive_verdict(True, True, False, True) == "inconsistent_evidence"
        assert conjunctive_verdict(False, True, False, True) == "negative"

    def test_a7_frozen_selfcheck_cases(self):
        """Frozen C6 self-check v2 (machine-verified): two scenarios."""
        assert a7_from_summary(0.03, 0.25, 0.12, 0.05) is True
        assert a7_from_summary(-0.01, 0.25, 0.12, 0.05) is False
        assert a7_from_summary(0.01, 0.03, 0.02, 0.05) == \
            "statistically_significant_below_substantive_threshold"

    def test_a7_refuses_uncalibrated_delta_min(self):
        """Frozen rule: no A7 without calibration."""
        cfg = TheOneConfig(delta_min=None)
        with pytest.raises(CalibrationRequiredError):
            a7_judgment(np.zeros((4, 2)), {1: np.ones(3), 0: np.zeros(3)}, cfg)

    def test_a7_full_pipeline_detects_confounding(self):
        """Binary T1-style data: strong confounding -> A7 True."""
        rng = np.random.default_rng(0)
        n = 4000
        u = rng.binomial(1, 0.5, n)
        x = rng.binomial(1, np.where(u == 1, 0.8, 0.2))
        py = np.select([(x == 1) & (u == 1), (x == 1) & (u == 0),
                        (x == 0) & (u == 1)], [0.9, 0.5, 0.6], default=0.2)
        y = rng.binomial(1, py)
        obs = np.column_stack([x, y]).astype(float)
        u2 = rng.binomial(1, 0.5, n)
        y1 = rng.binomial(1, np.where(u2 == 1, 0.9, 0.5)).astype(float)
        y0 = rng.binomial(1, np.where(u2 == 1, 0.6, 0.2)).astype(float)
        cfg = TheOneConfig(delta_min=0.05, bootstrap_B=400, seed=1)
        res = a7_judgment(obs, {1: y1, 0: y0}, cfg)
        assert res.are_different is True
        assert res.stats["tau_hat"] == pytest.approx(0.24, abs=0.06)


# ---------- CC-09 runner ----------
class TestRunner:
    def test_two_phase_and_burned_isolation(self, tmp_path):
        grid = {"beta_ux": [0.3, 0.6], "beta_uy": [0.4], "beta_xy": [0.2],
                "noise": [0.5], "n_samples": 2000, "base_seed": 42}
        rep = run_calibration(grid, str(tmp_path))
        assert rep["delta_min_frozen"] >= 0.05
        assert (tmp_path / "burned_list.json").exists()
        # frozen phase on the SAME grid must hard-fail (burned isolation)
        with pytest.raises(RuntimeError, match="BURNED-SET VIOLATION"):
            run_frozen({"grid": grid, "delta_min": rep["delta_min_frozen"]},
                       str(tmp_path))
        # a disjoint grid passes
        grid2 = dict(grid, base_seed=999, beta_ux=[0.35, 0.55])
        rep2 = run_frozen({"grid": grid2, "delta_min": rep["delta_min_frozen"]},
                          str(tmp_path))
        assert rep2["verdict"] in ("effective", "inconsistent_evidence", "negative")
        assert "scoping_note" in rep2  # honest scoping is part of the contract


# ---------- Q-C7 frozen grid + backdoor method (batch 02) ----------
class TestQC7GridAndMethod:
    def test_backdoor_method_recovers_beta_xy_baseline_is_biased(self):
        """Method (adjust for U) recovers beta_xy; baseline (unadjusted) carries
        the full confounding bias = product (Q-C7-3 + Jack self-check #1)."""
        spec = SCMSpec("linear_gaussian_3node", 0.6, 0.5, 0.3, 0.3, 200_000, 7)
        scm = SCMGenerator().generate(spec)
        adj = _backdoor_adjusted_int_estimate(scm)
        base = _assoc_int_estimate(scm)
        assert adj == pytest.approx(0.30, abs=0.02)   # == true int ATE
        assert base == pytest.approx(0.60, abs=0.02)  # == obs slope (biased)

    def test_backdoor_method_no_false_effect_when_beta_xy_zero(self):
        """Jack self-check #2: beta_xy=0, p=0.30. Method must NOT invent an effect;
        baseline falsely reports ~0.30 (false positive controlled by adjustment)."""
        b = (0.30) ** 0.5
        spec = SCMSpec("linear_gaussian_3node", b, b, 0.0, 0.3, 200_000, 9)
        scm = SCMGenerator().generate(spec)
        assert _backdoor_adjusted_int_estimate(scm) == pytest.approx(0.0, abs=0.02)
        assert _assoc_int_estimate(scm) == pytest.approx(0.30, abs=0.02)

    def test_grid_qc7_expansion_counts_and_decomposition(self):
        cfg = {"beta_xy": [0.0, 0.1, 0.3, 0.5],
               "products": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
               "noise": [0.1, 0.3, 0.5], "instances_per_cell": 50,
               "n_samples": 500, "base_seed": 1000}
        specs = list(SCMGenerator().grid_qc7(cfg))
        assert len(specs) == 4 * 9 * 3 * 50           # 5400
        assert len({s.seed for s in specs}) == 5400    # globally unique seeds
        for s in specs:                                # symmetric sqrt decomposition
            assert s.beta_ux == pytest.approx(s.beta_uy)
            assert s.beta_ux ** 2 == pytest.approx(s.beta_ux * s.beta_uy)

    def test_method_recovers_across_strengths_no_zero_bias(self, tmp_path):
        """Jack attack #4 lock-in: method must recover beta_xy at EVERY strength
        (no conservative zero-bias) and EG primary metric is median (attack #2)."""
        cal = {"beta_xy": [0.2], "products": [0.3], "noise": [0.3],
               "instances_per_cell": 5, "n_samples": 800, "base_seed": 100}
        run_calibration(cal, str(tmp_path))
        frozen = {"beta_xy": [0.0, 0.5], "products": [0.3], "noise": [0.3],
                  "instances_per_cell": 8, "n_samples": 800, "base_seed": 90000}
        rep = run_frozen({"grid": frozen, "delta_min": 0.1}, str(tmp_path))
        acc = rep["per_beta_xy_method_accuracy"]
        assert abs(acc["0.000"]["method_bias"]) < 0.05   # no false effect
        assert abs(acc["0.500"]["method_bias"]) < 0.05   # not suppressed toward 0
        egd = rep["eg_distribution"]
        assert egd["primary_metric"] == "median_and_quantiles"
        assert "mean_AUXILIARY_extreme_skewed_not_for_conclusion" in egd  # mean demoted

    def test_qc7_near_zero_cell_judged_by_absolute_error(self, tmp_path):
        """A beta_xy=0 cell triggers Amendment 1: excluded from EG ratio, judged on
        absolute error; method should beat baseline there."""
        cal = {"beta_xy": [0.3], "products": [0.2, 0.4], "noise": [0.3],
               "instances_per_cell": 5, "n_samples": 800, "base_seed": 1}
        run_calibration(cal, str(tmp_path))
        frozen = {"beta_xy": [0.0], "products": [0.2, 0.4], "noise": [0.3],
                  "instances_per_cell": 5, "n_samples": 800, "base_seed": 7000}
        rep = run_frozen({"grid": frozen, "delta_min": 0.1}, str(tmp_path))
        assert rep["near_zero_regime"] is not None
        assert rep["near_zero_regime"]["n"] == 10
        assert rep["near_zero_regime"]["method_better"] is True


# ---------- CC-10 memory ----------
class TestMemory:
    def test_persistence_provenance_delete_export(self, tmp_path):
        db = str(tmp_path / "m.db")
        s = MemoryStore(db)
        with pytest.raises(ValueError):
            s.put("k", "v", source="")          # provenance mandatory
        mid = s.put("project.name", "The One", source="user", ts=123.0)
        s.close()
        s2 = MemoryStore(db)                    # cross-connection persistence
        rec = s2.get(mid)
        assert rec["value"] == "The One" and rec["source"] == "user" and rec["ts"] == 123.0
        assert s2.search("project")[0]["id"] == mid
        line = json.loads(s2.export().splitlines()[0])
        assert line["key"] == "project.name"
        assert s2.delete(mid) is True
        assert s2.get(mid) is None              # delete means gone


# ---------- CC-11 agent ----------
class TestAgent:
    def test_routing_and_real_credentials(self, tmp_path):
        eng = InterventionEngine(confounded_graph())
        agent = Orchestrator(eng, MemoryStore(str(tmp_path / "m.db")))
        r1 = agent.handle("P(Y=1|do(X=1))")
        assert "0.700000" in r1.answer
        assert r1.credential["adjustment_set"] == ["U"]
        assert r1.credential["graph_hash"] == eng.g.content_hash()  # recomputable
        r2 = agent.handle("P(Y=1|X=1)")
        assert "0.820000" in r2.answer
        r3 = agent.handle("remember beachhead=medical")
        assert "stored" in r3.answer
        r4 = agent.handle("recall beachhead")
        assert "medical" in r4.answer
        r5 = agent.handle("write me a poem")
        assert r5.credential["method"] == "unrouted"


# ---------- CC-12 CLI ----------
class TestCLI:
    def test_demo_outputs_frozen_truth(self, capsys):
        assert cli_main(["demo", "causal"]) == 0
        out = capsys.readouterr().out
        assert "0.820000" in out and "0.700000" in out and "True" in out
        assert "graph_hash" in out

    def test_bench_calibrate_toy(self, tmp_path, capsys):
        assert cli_main(["bench", "mvp2a", "--phase", "calibrate",
                         "--out", str(tmp_path)]) == 0
        assert "delta_min_frozen" in capsys.readouterr().out
