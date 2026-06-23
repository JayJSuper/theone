"""Unit tests for the A-line productization shell: L0 env (detect/fallback),
L7 planning (tool router / dialogue state machine), L9 gateway."""
from __future__ import annotations
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


# ---- L0 detect ---------------------------------------------------------------
def test_detect_hardware_reports_real_machine():
    from theone.layer0_env.detect import detect_hardware, HardwareInfo
    hw = detect_hardware()
    assert isinstance(hw, HardwareInfo)
    assert hw.cpu_cores is None or hw.cpu_cores >= 1
    assert isinstance(hw.gpus, list)
    assert hw.has_gpu == (len(hw.gpus) > 0)
    d = hw.as_dict()
    assert "has_gpu" in d and "platform" in d


# ---- L0 fallback -------------------------------------------------------------
def test_choose_policy_gpu_and_cpu():
    from theone.layer0_env.detect import HardwareInfo, GPUInfo
    from theone.layer0_env.fallback import choose_policy
    gpu_hw = HardwareInfo("Linux x86_64", 32, 128.0, 500.0, [GPUInfo("H100", 80000)])
    p = choose_policy(gpu_hw)
    assert p.device == "cuda" and p.use_mamba is True and p.max_workers == 31

    cpu_hw = HardwareInfo("Darwin arm64", 8, 16.0, 200.0, [])
    p2 = choose_policy(cpu_hw)
    assert p2.device == "cpu" and p2.use_mamba is False and p2.max_workers == 7
    assert "CPU" in p2.reason or "cpu" in p2.reason


def test_choose_policy_handles_unknown_cores():
    from theone.layer0_env.detect import HardwareInfo
    from theone.layer0_env.fallback import choose_policy
    p = choose_policy(HardwareInfo("x", None, None, None, []))
    assert p.device == "cpu" and p.max_workers >= 1


# ---- L7 tool router ----------------------------------------------------------
@pytest.mark.parametrize("text,intent,verifiable", [
    ("what is the effect of X on Y?", "causal_query", True),
    ("compute P(Y | do(X=1))", "causal_query", True),
    ("please remember that the launch is friday", "memory_op", True),
    ("recall what we said about pricing", "memory_op", True),
    ("write a python function to sort a list", "code", False),
    ("how are you today?", "chat", False),
])
def test_router_classifies_and_marks_provenance(text, intent, verifiable):
    from theone.layer7_planning.tool_router import ToolRouter
    r = ToolRouter().route(text)
    assert r.intent.value == intent
    assert r.verifiable is verifiable


def test_router_memory_imperative_beats_causal_payload():
    from theone.layer7_planning.tool_router import ToolRouter
    # "remember that X causes Y" is a store op (verified at store time), not a query
    r = ToolRouter().route("remember that smoking causes cancer")
    assert r.intent.value == "memory_op"


# ---- L7 dialogue state machine ----------------------------------------------
def test_dialogue_lifecycle_and_verified_ratio():
    from theone.layer7_planning.dialogue_state import DialogueStateMachine, DialogueState
    dsm = DialogueStateMachine("s1")
    assert dsm.session.verified_ratio == 0.0
    t = dsm.receive("effect of X on Y?")
    assert dsm.session.state == DialogueState.ROUTED and t.route.verifiable
    dsm.record_answer("0.66", verified=True)
    assert dsm.session.state == DialogueState.ANSWERED
    dsm.receive("tell me a joke")
    dsm.record_answer("haha", verified=False)
    assert len(dsm.history()) == 2
    assert dsm.session.verified_ratio == 0.5


def test_dialogue_record_without_turn_raises():
    from theone.layer7_planning.dialogue_state import DialogueStateMachine
    with pytest.raises(RuntimeError):
        DialogueStateMachine().record_answer("x", verified=True)


# ---- L9 gateway --------------------------------------------------------------
def test_gateway_message_provenance():
    from theone.layer9_app.gateway import TheOneGateway
    gw = TheOneGateway()
    assert gw.handle_message("effect of X on Y")["verifiable"] is True
    assert gw.handle_message("write code")["verifiable"] is False
    assert gw.health()["status"] == "ok"


def test_gateway_causal_answer_carries_recomputed_credential():
    from theone.layer9_app.gateway import TheOneGateway
    out = TheOneGateway().handle_causal(confounded_fork(), "X", "Y")
    assert out["decision"] == "answer"
    assert out["recomputed_ok"] is True
    assert out["verifiable"] is True
    assert abs(out["value"] - 0.66) < 1e-9
    assert "sensitivity" in out["evidence"]


def test_gateway_verify_credential_match_and_mismatch():
    from theone.layer9_app.gateway import TheOneGateway
    assert TheOneGateway.verify_credential(0.703, 0.703)["verified"] is True
    bad = TheOneGateway.verify_credential(0.703, 0.650)
    assert bad["verified"] is False and bad["gap"] > 0.05
    # non-numeric falls back to equality
    assert TheOneGateway.verify_credential("abc", "abc")["verified"] is True


def test_create_app_fails_cleanly_without_fastapi():
    from theone.layer9_app.gateway import create_app
    try:
        import fastapi  # noqa: F401
        app = create_app()
        assert app is not None          # fastapi present -> builds an app
    except ImportError:
        with pytest.raises(RuntimeError, match="FastAPI"):
            create_app()                 # absent -> clear, non-crashing error
