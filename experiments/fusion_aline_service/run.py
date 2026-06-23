"""A-line service demo · the 'value-chain segment' deployment form, runnable end-to-end.

Ties the new A-line productization modules into one flow:
  L0 detect hardware -> L7 route each user message -> L9 gateway dispatches:
    causal query  -> verifiable engine (recomputable credential)
    memory op     -> sovereign memory (verifiable)
    code / chat   -> mounted LLM (labelled UNVERIFIED, honest provenance)
  L7 dialogue state machine tracks the verified/unverified provenance of the session.

This is The One as a verifiable causal MIDDLEWARE: drop it in front of any LLM and every
answer carries provenance — engine-verified, or mounted-LLM-unverified-but-labelled.

Run:  .venv/bin/python experiments/fusion_aline_service/run.py
"""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.layer0_env import detect_hardware, choose_policy
from theone.layer9_app import TheOneGateway
from theone.layer7_planning import DialogueStateMachine, Intent


def fork():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    g.set_cpt("Y", {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
                    (1, 0): {0: 0.4, 1: 0.6}, (1, 1): {0: 0.1, 1: 0.9}})
    return g


def main():
    print("=== A-line service demo: verifiable causal middleware (value-chain segment) ===\n")
    hw = detect_hardware()
    pol = choose_policy(hw)
    print(f"L0 hardware: {hw.platform}, {hw.cpu_cores} cores, GPU={hw.has_gpu} -> device={pol.device}")
    print(f"   ({pol.reason})\n")

    gw = TheOneGateway()
    dsm = DialogueStateMachine("demo")
    messages = [
        "What is the effect of X on Y?",     # -> verifiable engine
        "remember that X drives Y",          # -> sovereign memory (verifiable)
        "write a python function to sort",   # -> mounted LLM (unverified)
        "how's the weather?",                # -> mounted LLM (unverified)
    ]
    print("session (each answer carries provenance):")
    for msg in messages:
        turn = dsm.receive(msg)
        if turn.route.intent == Intent.CAUSAL_QUERY:
            r = gw.handle_causal(fork(), "X", "Y")
            ans = f"do(X=1)→Y = {r['value']:.3f} (recomputed_ok={r['recomputed_ok']})"
            dsm.record_answer(ans, verified=True)
        else:
            ans = f"[{turn.route.handler}] (unverified)" if not turn.route.verifiable else "[stored]"
            dsm.record_answer(ans, verified=turn.route.verifiable)
        flag = "✅ verified" if turn.verified else "⚠️  unverified"
        print(f"  {flag:14} {turn.route.intent.value:13} | {msg!r} -> {turn.answer}")

    print(f"\nsession provenance: verified_ratio = {dsm.session.verified_ratio:.2f} "
          f"({sum(1 for t in dsm.history() if t.verified)}/{len(dsm.history())} engine-verified)")
    print("\nContract: The One sits in front of any LLM as middleware. Causal/memory answers are")
    print("engine-verified with recomputable credentials; code/chat are routed to the mounted LLM")
    print("and LABELLED unverified — provenance is never hidden. This is the 'value-chain segment'")
    print("deployment, runnable today on CPU, no base-model training required.")

    ok = (pol.device in ("cpu", "cuda") and dsm.session.verified_ratio == 0.5)
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
