"""Fusion deepening⑦ · L1 LLM adapter — 'LLM proposes, The One verifies'.

The LLM enters as a perception organ: its natural-language causal claim is PARSED into
a structured candidate (never trusted), then VERIFIED against the engine's exact do().
  • well-formed claim -> high-confidence candidate; vague text -> not actionable.
  • a correct claim is corroborated; a hallucinated number is REFUTED (caught), because
    trust comes from the engine recomputation, not from the LLM's fluency.

Run:  .venv/bin/python experiments/fusion_llm_adapter/run.py
"""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.layer1_perception import LLMAdapter


def chain_ate03():
    """X -> Y with interventional ATE = P(Y|do X=1) - P(Y|do X=0) = 0.8 - 0.5 = 0.30."""
    g = CausalGraph()
    for n in ("X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("X", "Y")
    g.set_cpt("X", {(): {0: 0.5, 1: 0.5}})
    g.set_cpt("Y", {(0,): {0: 0.5, 1: 0.5}, (1,): {0: 0.2, 1: 0.8}})
    return g


def main():
    print("=== Fusion deepening⑦: L1 LLM adapter ('LLM proposes, The One verifies') ===\n")
    ad = LLMAdapter()
    ok = True

    print("parsing LLM outputs into structured candidates:")
    samples = [
        "The effect of X on Y is 0.30, adjusting for U.",
        "X causes Y.",
        "Honestly it's complicated and many factors interact.",
    ]
    parsed = [ad.parse(s) for s in samples]
    for c in parsed:
        print(f"   {c.treatment}->{c.target} effect={c.effect} adj={c.adjustment_set} "
              f"conf={c.confidence} actionable={c.is_actionable()}")
    ok &= parsed[0].is_actionable() and parsed[0].effect == 0.30 and parsed[0].adjustment_set == ["U"]
    ok &= parsed[1].is_actionable() and parsed[1].effect is None
    ok &= not parsed[2].is_actionable()

    print("\nLLM proposes, The One verifies (engine ATE = 0.30):")
    ate = InterventionEngine(chain_ate03()).interventional_ate("X", "Y")
    good = ad.parse("the effect of X on Y is 0.30")
    bad = ad.parse("the effect of X on Y is 0.60")
    vg = LLMAdapter.verify_against_engine(good, ate)
    vb = LLMAdapter.verify_against_engine(bad, ate)
    print(f"   LLM says 0.30 -> {vg['verdict']} (engine {vg['engine_effect']}, gap {vg['gap']})")
    print(f"   LLM says 0.60 -> {vb['verdict']} (engine {vb['engine_effect']}, gap {vb['gap']}) "
          f"<- hallucinated number caught")
    ok &= vg["verdict"] == "corroborated" and vb["verdict"] == "refuted"

    print("\nLLM-adapter contract: the LLM is a perception organ, not an oracle. Its claim is")
    print("parsed into a candidate and only believed when the engine's exact do() corroborates")
    print("it; a confident wrong number is refuted by recomputation. This is the positioning made")
    print("concrete — LLM for fluency/coverage, The One for verifiable causal truth.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
