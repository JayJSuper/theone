"""Fusion capstone · the cognitive-OS loop: perceive -> verify -> remember -> act.

The whole 'super form' running one complete cognitive cycle, with the credential spine
and abstain bus protecting every step against a hallucinated input:

  1. PERCEIVE (L1 LLM adapter): an LLM's causal claim is parsed into a candidate.
  2. VERIFY   (L2 causal engine): the engine computes the exact interventional effect
     and corroborates or REFUTES the claim. A refuted claim stops here.
  3. REMEMBER (L4 sovereign memory): only a verified claim is stored, by causal signature.
  4. ACT      (L5 execution): an action is gated on the verified causal credential;
     without corroboration the causal gate is inadmissible and the action ABSTAINS.

Two inputs: a CORRECT claim flows all the way to a remembered + executed action; a
HALLUCINATED claim is stopped at verification — never stored, never acted upon.

Run:  .venv/bin/python experiments/fusion_cognitive_os/run.py
"""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.layer1_perception import LLMAdapter
from theone.layer4_memory import MemoryLayer
from theone.layer5_execution import ExecutionLayer


def world():
    """Ground-truth world the engine reasons over: X -> Y with ATE 0.30."""
    g = CausalGraph()
    for n in ("X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("X", "Y")
    g.set_cpt("X", {(): {0: 0.5, 1: 0.5}})
    g.set_cpt("Y", {(0,): {0: 0.5, 1: 0.5}, (1,): {0: 0.2, 1: 0.8}})
    return g


def cognitive_cycle(llm_text, mem, engine, ate, label):
    print(f"--- input: {llm_text!r}")
    adapter = LLMAdapter()
    # 1. PERCEIVE
    claim = adapter.parse(llm_text)
    print(f"  1. PERCEIVE  -> {claim.treatment}->{claim.target} effect={claim.effect} conf={claim.confidence}")
    if not claim.is_actionable() or claim.effect is None:
        print("     not actionable -> ABSTAIN (no verifiable candidate)\n")
        return "abstain_perceive"
    # 2. VERIFY
    v = LLMAdapter.verify_against_engine(claim, ate)
    print(f"  2. VERIFY    -> engine ATE={v['engine_effect']} | claim={v['claim_effect']} | {v['verdict']}")
    if v["verdict"] != "corroborated":
        print("     refuted by recomputation -> STOP (not stored, not acted)\n")
        return "refuted"
    # 3. REMEMBER (verified claim only)
    cred = {"treatment": claim.treatment, "target": claim.target,
            "adjustment_set": claim.adjustment_set, "effect": round(ate, 4), "regime": "verified"}
    rv = mem.run({"op": "remember", "text": llm_text, "credential": cred})
    print(f"  3. REMEMBER  -> stored by signature {rv.credential.value}")
    # 4. ACT (gated on the verified causal credential)
    ex = ExecutionLayer(sandbox_root="/tmp")
    causal_cred = {"recomputable": True, "admissible": True}   # corroborated => recomputable+admissible
    av = ex.run({"action_kind": "write", "target": "/tmp/theone_action.txt",
                 "content": f"act on verified {claim.treatment}->{claim.target}",
                 "causal_credential": causal_cred})
    print(f"  4. ACT       -> {'EXECUTE-cleared' if av.is_answer() else 'ABSTAIN'}\n")
    return "completed" if av.is_answer() else "abstain_act"


def main():
    print("=== Fusion capstone: the cognitive-OS loop (perceive->verify->remember->act) ===\n")
    engine = InterventionEngine(world())
    ate = engine.interventional_ate("X", "Y")
    mem = MemoryLayer(":memory:")

    r1 = cognitive_cycle("The effect of X on Y is 0.30.", mem, engine, ate, "correct")
    r2 = cognitive_cycle("Actually the effect of X on Y is 0.65.", mem, engine, ate, "hallucinated")

    # the correct claim completed the full loop; the hallucinated one was refuted
    n_stored = len(mem.mem._all_live())
    mem.close()
    ok = (r1 == "completed" and r2 == "refuted" and n_stored == 1)
    print("Capstone contract: a verified causal claim flows perceive->verify->remember->act;")
    print("a hallucinated claim is refuted at verification and never reaches memory or action.")
    print(f"  memories stored = {n_stored} (only the verified claim)")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
