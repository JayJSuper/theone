"""Unit tests for the six fusion layers + L2 deepening legs."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from theone.types import Variable
from theone.causal.graph import CausalGraph


# ---- fixtures ----------------------------------------------------------------
def confounded_fork():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    g.set_cpt("Y", {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
                    (1, 0): {0: 0.4, 1: 0.6}, (1, 1): {0: 0.1, 1: 0.9}})
    return g


def fork_df(n, seed):
    g = confounded_fork()
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.4).astype(int)
    x = (rng.random(n) < np.where(u == 1, 0.7, 0.2)).astype(int)
    py = {(0, 0): .1, (0, 1): .5, (1, 0): .6, (1, 1): .9}
    y = np.array([1 if rng.random() < py[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    return pd.DataFrame({"U": u, "X": x, "Y": y})


# ---- L0 physics --------------------------------------------------------------
def test_l0_physics_answer_and_abstain():
    from theone.layer0_physics import PhysicsLayer
    assert PhysicsLayer().run({"q0": [1.0], "p0": [0.0], "steps": 5000}).is_answer()
    assert not PhysicsLayer(omega=5.0, dt=0.2).run({"q0": [1.0], "p0": [0.0], "steps": 5000}).is_answer()


# ---- L1 perception -----------------------------------------------------------
def test_l1_perception_sine_answers_nan_abstains():
    from theone.layer1_perception import PerceptionLayer
    t = np.linspace(0, 10, 500)
    L = PerceptionLayer(hidden_dim=32, seed=1)
    v = L.run({"signal": np.sin(2 * np.pi * t)})
    assert v.is_answer() and v.credential.value < 1e-3
    bad = np.sin(2 * np.pi * t).copy(); bad[5] = np.nan
    assert not L.run({"signal": bad}).is_answer()
    assert not PerceptionLayer(spectral_radius=1.0).run({"signal": np.sin(t)}).is_answer()


# ---- L2 causal + IPRG + E-value ----------------------------------------------
def test_l2_causal_answers_with_iprg_and_evalue():
    from theone.layer2_world_model import CausalLayer
    v = CausalLayer().run({"graph": confounded_fork(), "treatment": "X", "target": "Y"})
    assert v.is_answer()
    assert abs(v.credential.value - 0.66) < 1e-9
    assert "sensitivity" in v.credential.evidence
    assert v.credential.evidence["sensitivity"]["e_value"] > 1.0


def test_l2_continuous_do_answer_and_abstain():
    from theone.layer2_world_model import ContinuousCausalLayer
    rng = np.random.default_rng(3)
    n, p = 30000, 8
    u = rng.standard_normal(n)
    a = rng.uniform(0.6, 1.4, p) * rng.choice([-1.0, 1.0], p)
    P = u[:, None] * a[None, :] + rng.normal(0, 0.3, (n, p))   # clean proxies -> low bias/variance
    x = (rng.random(n) < 1 / (1 + np.exp(-u))).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * x + 1.8 * u)))).astype(float)
    L = ContinuousCausalLayer()
    assert L.run({"proxies": P, "treatment": x, "outcome": y}).is_answer()
    # single proxy -> completeness uncheckable -> abstain
    assert not L.run({"proxies": P[:, :1], "treatment": x, "outcome": y}).is_answer()


def test_l2_discovery_answer_and_abstain():
    from theone.layer2_world_model import CausalDiscoveryLayer
    L = CausalDiscoveryLayer()
    assert L.run({"data": fork_df(1500, 0), "B": 12, "seed": 1}).is_answer()
    assert not L.run({"data": fork_df(25, 2), "B": 12, "seed": 1}).is_answer()


# ---- L3 decision + cognitive updater -----------------------------------------
def test_l3_decision_answer_and_abstain():
    from theone.layer3_decision import DecisionLayer
    rng = np.random.default_rng(0)
    W = rng.standard_normal((8, 4))
    o = W @ (rng.standard_normal(4) * 0.3)
    o = o / (np.linalg.norm(o) + 1e-9) * np.sqrt(2.0)
    assert DecisionLayer().run({"W": W, "observation": o}).is_answer()
    assert not DecisionLayer().run({"W": W, "observation": o, "iters": 1}).is_answer()


def test_l3_cognitive_updater_shift_and_stationary():
    from theone.layer3_decision import CognitiveUpdater
    rng = np.random.default_rng(0)
    n = 2000
    x = (rng.random(n) < 0.5).astype(int); z = (rng.random(n) < 0.5).astype(int)
    y_zy = (rng.random(n) < np.where(z == 1, 0.8, 0.2)).astype(int)
    shift = pd.DataFrame({"X": x, "Z": z, "Y": y_zy})
    y_xy = (rng.random(n) < np.where(x == 1, 0.8, 0.2)).astype(int)
    stat = pd.DataFrame({"X": x, "Z": z, "Y": y_xy})
    U = CognitiveUpdater()
    assert U.run({"old_edges": [("X", "Y")], "data": shift}).is_answer()
    assert not U.run({"old_edges": [("X", "Y")], "data": stat}).is_answer()


# ---- L4 memory + legs --------------------------------------------------------
def test_l4_memory_recall_and_abstain():
    from theone.layer4_memory import MemoryLayer
    from theone.memory.signature import CausalSignature
    mem = MemoryLayer(":memory:")
    cred = {"treatment": "X", "target": "Y", "adjustment_set": ["U"], "effect": 0.66, "regime": "normal"}
    mem.run({"op": "remember", "text": "m", "credential": cred})
    q = CausalSignature.from_credential(cred)
    assert mem.run({"op": "recall", "query_signature": q}).is_answer()
    far = CausalSignature("X", "Y", ("U",), 0.66, "crisis")
    assert not mem.run({"op": "recall", "query_signature": far}).is_answer()
    mem.close()


def test_l4_pattern_and_conflict():
    from theone.memory.sovereign import SovereignMemory
    from theone.layer4_memory import PatternRecognizer, ConflictArbitrator
    mem = SovereignMemory(":memory:")
    c = lambda e, r: {"treatment": "X", "target": "Y", "adjustment_set": ["U"], "effect": e, "regime": r}
    mem.remember("a", c(0.66, "normal"), "s1")
    mem.remember("b", c(0.45, "normal"), "s2")
    mem.remember("c", c(0.30, "stressed"), "s3")
    assert any(p["edge"] == ["X", "Y"] and p["count"] == 3
               for p in PatternRecognizer(mem).frequent_edges(min_support=2))
    conflicts = ConflictArbitrator(mem).find_conflicts(effect_tol=0.1)
    assert len(conflicts) == 1 and conflicts[0]["question"].endswith("regime=normal")
    mem.close()


# ---- L5 execution ------------------------------------------------------------
def test_l5_execution_safe_and_dangerous():
    from theone.layer5_execution import ExecutionLayer
    ex = ExecutionLayer(sandbox_root="/tmp")
    assert ex.run({"action_kind": "command", "command": "echo hi"}).is_answer()
    assert not ex.run({"action_kind": "command", "command": "rm -rf /"}).is_answer()
    assert not ex.run({"action_kind": "write", "target": "/tmp/x", "content": "x",
                       "causal_credential": {"recomputable": False, "admissible": True}}).is_answer()


# ---- L0 PINN -----------------------------------------------------------------
def test_l0_pinn_extrapolation_benefit():
    from theone.layer0_physics import extrapolation_benefit
    r = extrapolation_benefit(omega=1.0, degree=12, seed=0)
    assert r["improvement"] > 0.5 and r["rmse_physics"] < 0.1


# ---- L1 LLM adapter ----------------------------------------------------------
def test_llm_adapter_parse_and_verify():
    from theone.layer1_perception import LLMAdapter
    ad = LLMAdapter()
    good = ad.parse("The effect of X on Y is 0.30, adjusting for U.")
    assert good.is_actionable() and good.effect == 0.30 and good.adjustment_set == ["U"]
    vague = ad.parse("it's complicated and many things interact")
    assert not vague.is_actionable()
    assert LLMAdapter.verify_against_engine(good, 0.30)["verdict"] == "corroborated"
    assert LLMAdapter.verify_against_engine(good, 0.60)["verdict"] == "refuted"


# ---- temporal causal direction -----------------------------------------------
def test_temporal_direction_recovers_and_abstains():
    from theone.layer2_world_model.temporal_causal import temporal_direction
    rng = np.random.default_rng(0)
    n = 3000
    a = np.zeros(n); b = np.zeros(n); ai = np.zeros(n); bi = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.5 * a[t - 1] + rng.normal()
        b[t] = 0.4 * b[t - 1] + 0.6 * a[t - 1] + rng.normal()
        ai[t] = 0.5 * ai[t - 1] + rng.normal()
        bi[t] = 0.5 * bi[t - 1] + rng.normal()
    assert temporal_direction(a, b)["verdict"] == "a->b"
    assert temporal_direction(ai, bi)["verdict"] == "abstain"


# ---- sensitivity -------------------------------------------------------------
def test_e_value_monotone_and_formula():
    from theone.layer2_world_model import e_value_for_do, e_value_rr
    assert e_value_for_do(0.8, 0.2)["e_value"] > e_value_for_do(0.55, 0.45)["e_value"]
    assert e_value_rr(1.0) == pytest.approx(1.0)
    assert e_value_rr(4.0) == pytest.approx(4.0 + (4.0 * 3.0) ** 0.5)
