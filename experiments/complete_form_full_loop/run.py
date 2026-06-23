"""Complete-form FULL cognitive-OS loop — perceive -> verify -> remember -> recall -> ACT.

Closes the loop: a verified causal conclusion is remembered, recalled for a new decision by
de-confounded signature, and turned into a CREDENTIALED ACTION — or an honest abstain. The
system never acts on an unverified or fragile belief.

  CASE A (strong, verifiable effect)  -> recommend an action, with the credential.
  CASE B (fragile effect, engine REJECTs) -> ABSTAIN, no action, honest reason.

Run:  .venv/bin/python experiments/complete_form_full_loop/run.py
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
    print("=== complete-form FULL loop · perceive -> verify -> remember -> recall -> act ===\n")
    cf = CompleteForm()
    mem = SovereignMemory(":memory:")

    # CASE A — strong effect: verify -> remember -> recall -> ACT
    rA = cf.analyze(make(8000, 0.30, 1), pre_treatment=["U", "Z"])
    mid = mem.remember("treatment effect (cohort A)", rA.credential, source="cf")
    q = CausalSignature(treatment="X", target="Y", adjustment_set=("U",),
                        effect=rA.effect, regime=rA.credential["regime"])
    recalled = mem.recall_for_decision(q, k=1)[0]
    actA = cf.recommend(rA)
    print(f"CASE A: verified effect {rA.effect:+.3f} -> remembered(id {mid}) -> recalled(id {recalled.mem_id})")
    print(f"  ACT: {actA['action'].upper()} — {actA.get('decision','')}")
    print(f"       because: {actA.get('because','')}")

    # CASE B — fragile effect: engine REJECTs -> ABSTAIN (no action)
    rB = cf.analyze(make(8000, 0.015, 3), pre_treatment=["U", "Z"])
    actB = cf.recommend(rB)
    print(f"\nCASE B: zone={rB.zone} effect {rB.effect:+.3f}")
    print(f"  ACT: {actB['action'].upper()} — {actB['reason']}")

    g1 = actA["action"] == "recommend" and "apply" in actA["decision"]   # acts on verified strong effect
    g2 = recalled.mem_id == mid                                          # recall closed before acting
    g3 = actB["action"] == "abstain"                                     # refuses to act on fragile belief
    g4 = rA.trustworthy and not rB.trustworthy                           # trust gate drives the act
    allok = g1 and g2 and g3 and g4
    print("\nfull-loop gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] strong verified effect -> credentialed ACTION")
    print(f"  [{'PASS' if g2 else 'FAIL'}] recall (signature) closed before acting")
    print(f"  [{'PASS' if g3 else 'FAIL'}] fragile effect -> ABSTAIN, no action")
    print(f"  [{'PASS' if g4 else 'FAIL'}] trust gate (zone+replay) drives act vs abstain")
    print(f"\n  >>> {'PASS — complete form closes perceive->verify->remember->recall->act, credentialed' if allok else 'CHECK'}")
    print("\nMeaning: the complete form is a full cognitive OS — it perceives, verifies, accumulates")
    print("beliefs, recalls the causally-right one, and ACTS only on what it can recompute; on a")
    print("fragile belief it abstains. Every action carries its credential. Honest: toy SCMs.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
