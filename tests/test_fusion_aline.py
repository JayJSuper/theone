"""Unit tests for the A-line productization modules (L0 env, L7 planning, L9 gateway)."""
from __future__ import annotations

from theone.types import Variable
from theone.causal.graph import CausalGraph


# ---- L0 hardware detect + fallback ------------------------------------------
def test_l0_detect_and_policy():
    from theone.layer0_env import detect_hardware, choose_policy
    hw = detect_hardware()
    assert hw.cpu_cores is None or hw.cpu_cores >= 1
    assert isinstance(hw.has_gpu, bool)
    p = choose_policy(hw)
    assert p.device in ("cuda", "cpu")
    assert p.device == ("cuda" if hw.has_gpu else "cpu")
    assert p.max_workers >= 1


# ---- L7 tool router + dialogue state ----------------------------------------
def test_l7_router_classifies():
    from theone.layer7_planning import ToolRouter, Intent
    r = ToolRouter()
    assert r.route("what is the effect of X on Y?").intent == Intent.CAUSAL_QUERY
    assert r.route("remember that X causes Y").intent == Intent.MEMORY_OP
    assert r.route("write a python function to sort").intent == Intent.CODE
    assert r.route("hello there").intent == Intent.CHAT
    assert r.route("do(X=1) on Y?").verifiable is True
    assert r.route("tell me a joke").verifiable is False


def test_l7_dialogue_tracks_provenance():
    from theone.layer7_planning import DialogueStateMachine, DialogueState
    dsm = DialogueStateMachine("s")
    t = dsm.receive("effect of X on Y?")
    assert dsm.session.state == DialogueState.ROUTED
    dsm.record_answer("0.66", verified=t.route.verifiable)
    dsm.receive("hi"); dsm.record_answer("hello", verified=False)
    assert dsm.session.state == DialogueState.ANSWERED
    assert dsm.session.verified_ratio == 0.5
    assert len(dsm.history()) == 2


# ---- L9 gateway -------------------------------------------------------------
def _fork():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    g.set_cpt("Y", {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
                    (1, 0): {0: 0.4, 1: 0.6}, (1, 1): {0: 0.1, 1: 0.9}})
    return g


def test_l9_gateway_message_and_causal():
    from theone.layer9_app import TheOneGateway
    gw = TheOneGateway()
    assert gw.health()["status"] == "ok"
    m = gw.handle_message("what is the effect of X on Y?")
    assert m["intent"] == "causal_query" and m["verifiable"] is True
    assert gw.handle_message("write code")["verifiable"] is False
    r = gw.handle_causal(_fork(), "X", "Y")
    assert r["decision"] == "answer" and r["recomputed_ok"] is True
    assert abs(r["value"] - 0.66) < 1e-9
    assert gw.verify_credential(0.66, 0.66)["verified"] is True
    assert gw.verify_credential(0.66, 0.99)["verified"] is False
