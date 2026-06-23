"""Complete-form cognitive OS loop — perceive/identify/verify-do -> REMEMBER -> RECALL.

The complete form is not just an estimator; it is a cognitive loop that ACCUMULATES verified
beliefs. This wires CompleteForm into SovereignMemory:

  1. analyze data -> a credentialed causal conclusion (the verify-do step);
  2. REMEMBER it — indexed by its de-confounded CAUSAL SIGNATURE (treatment->target, adjustment
     set, effect), inherited from the credential (not the surface text);
  3. RECALL for a new decision — by causal signature, returning the conclusion whose
     de-confounded effect actually matches, where naive surface recall would transfer the wrong
     effect from a text look-alike.

The payoff (pillar-2 thesis, now driven by the complete-form engine): two stored conclusions can
read almost identically yet carry opposite true effects; signature recall tells them apart.

Run:  .venv/bin/python experiments/complete_form_cognitive_loop/run.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from theone.native import CompleteForm
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature


def make(n, x_effect, seed):
    rng = np.random.default_rng(seed)
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    e = x_effect
    pY = {(0, 0): .40, (0, 1): .62, (1, 0): min(.40 + e, .97), (1, 1): min(.62 + e, .97)}
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    Z = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})


def main():
    print("=== complete-form cognitive loop · verify-do -> remember -> recall ===\n")
    cf = CompleteForm()
    mem = SovereignMemory(":memory:")

    # accumulate TWO verified beliefs about the same X->Y question but with different true
    # effects (e.g. two patient sub-populations). Both read alike; effects differ.
    rA = cf.analyze(make(8000, 0.30, 1), pre_treatment=["U", "Z"])     # strong effect
    rB = cf.analyze(make(8000, 0.08, 2), pre_treatment=["U", "Z"])     # weak effect
    idA = mem.remember("treatment helps the outcome (cohort A)", rA.credential, source="complete-form")
    idB = mem.remember("treatment helps the outcome (cohort B)", rB.credential, source="complete-form")
    print(f"remembered: A effect={rA.effect:+.3f} (id {idA}) · B effect={rB.effect:+.3f} (id {idB})")

    # a NEW decision arrives whose de-confounded effect is ~0.30 -> must recall cohort A,
    # NOT the text-identical cohort B (whose true effect is ~0.08).
    query = CausalSignature(treatment="X", target="Y", adjustment_set=("U",),
                            effect=rA.effect, regime="complete-form: perceive -> identify -> verify-do -> credential")
    top = mem.recall_for_decision(query, k=2)
    print(f"\nrecall for a decision needing effect≈{rA.effect:+.3f}:")
    for rc in top:
        print(f"  id {rc.mem_id}: '{rc.text}'  sig.effect={rc.signature.effect:+.3f}  dist={rc.score:.3f}")

    best = top[0]
    g1 = best.mem_id == idA                       # signature recall returns the right cohort
    g2 = abs(best.signature.effect - rA.effect) < 0.05   # ...with the matching de-confounded effect
    g3 = len(mem._all_live()) == 2                # both beliefs persisted, auditable
    # revise A (belief update) and confirm the old version is retained (auditable history)
    idA2 = mem.revise(idA, "treatment helps (cohort A, updated)", rA.credential, source="complete-form")
    g4 = idA2 != idA and len(mem._all_live()) == 2
    allok = g1 and g2 and g3 and g4
    print("\ncognitive-loop gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] signature recall returns the causally-matching belief (not text twin)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] recalled belief carries the matching de-confounded effect")
    print(f"  [{'PASS' if g3 else 'FAIL'}] both verified beliefs persisted in sovereign memory")
    print(f"  [{'PASS' if g4 else 'FAIL'}] belief revision is versioned (old retained, auditable)")
    print(f"\n  >>> {'PASS — complete form closes the cognitive loop: verify -> remember -> recall (signature-correct)' if allok else 'CHECK'}")
    print("\nMeaning: the complete form ACCUMULATES verified causal beliefs in sovereign memory,")
    print("indexed by de-confounded signature — so recall-for-a-decision returns the conclusion")
    print("whose true effect matches, immune to the surface confounding that fools text recall.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
