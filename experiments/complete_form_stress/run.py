"""Complete-form STRESS test — the integrity red-lines hold across many random regimes.

Before any product push, the complete form's HONESTY must hold under adversarial variety, not
just hand-picked demos. We sweep many random SCMs spanning strong / weak / null effects and
verify the invariants that make The One trustworthy:

  R1  a NULL / near-zero true effect is NEVER acted on (recommend) — it abstains or stays small;
  R2  a STRONG identifiable effect IS recovered (low bias) and acted on;
  R3  EVERY verified conclusion is replay-checked (no un-recomputable claim);
  R4  recall-by-signature is always causally correct (never returns a wrong-effect twin);
  R5  no crash across the regime sweep.

A single red-line violation (acting on a null effect) fails the suite — that is the property a
product must never break.

Run:  .venv/bin/python experiments/complete_form_stress/run.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from theone.native import CompleteForm
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature


def make(n, x_effect, conf_strength, seed):
    rng = np.random.default_rng(seed)
    pu = float(rng.uniform(0.3, 0.6))
    U = (rng.random(n) < pu).astype(int)
    px = np.where(U == 1, 0.5 + conf_strength, 0.5 - conf_strength)
    X = (rng.random(n) < np.clip(px, 0.05, 0.95)).astype(int)
    base0, base1 = rng.uniform(0.25, 0.45), rng.uniform(0.55, 0.7)   # U effect
    e = x_effect
    pY = {(0, 0): base0, (0, 1): base1,
          (1, 0): min(base0 + e, .97), (1, 1): min(base1 + e, .97)}
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    Z = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})


def main():
    print("=== complete-form STRESS · integrity red-lines across random regimes ===\n")
    cf = CompleteForm()
    rng = np.random.default_rng(0)
    N = 40

    null_acted = 0          # R1 violations: acted on a near-zero true effect
    strong_ok = 0; strong_total = 0
    replay_all = True
    crashes = 0
    recall_ok = True
    null_total = 0

    for i in range(N):
        kind = ["null", "weak", "strong"][i % 3]
        x_eff = {"null": 0.0, "weak": rng.uniform(0.03, 0.08), "strong": rng.uniform(0.25, 0.35)}[kind]
        conf = float(rng.uniform(0.15, 0.3))
        try:
            r = cf.analyze(make(7000, x_eff, conf, seed=100 + i), pre_treatment=["U", "Z"])
            act = cf.recommend(r)
        except Exception as e:
            crashes += 1; print(f"  [{i}] {kind}: CRASH {type(e).__name__}"); continue
        if r.credential.get("replay_ok") is not True:
            replay_all = False
        if kind == "null":
            null_total += 1
            if act["action"] == "recommend":
                null_acted += 1
                print(f"  [{i}] NULL but ACTED (red-line!) effect={r.effect:+.3f} zone={r.zone}")
        if kind == "strong":
            strong_total += 1
            if abs(r.effect - x_eff) < 0.06 and act["action"] == "recommend":
                strong_ok += 1

    # R4: signature recall correctness on two stored beliefs with opposite effects
    mem = SovereignMemory(":memory:")
    rS = cf.analyze(make(8000, 0.30, 0.25, 1), pre_treatment=["U", "Z"])
    rW = cf.analyze(make(8000, 0.05, 0.25, 2), pre_treatment=["U", "Z"])
    idS = mem.remember("strong", rS.credential, source="s")
    mem.remember("weak", rW.credential, source="w")
    q = CausalSignature(treatment="X", target="Y", adjustment_set=("U",),
                        effect=rS.effect, regime=rS.credential["regime"])
    recall_ok = mem.recall_for_decision(q, k=1)[0].mem_id == idS

    print(f"\nregimes run: {N}   null acted-on: {null_acted}/{null_total}   "
          f"strong recovered+acted: {strong_ok}/{strong_total}   crashes: {crashes}")

    R1 = null_acted == 0
    R2 = strong_ok >= 0.7 * max(strong_total, 1)
    R3 = replay_all
    R4 = recall_ok
    R5 = crashes == 0
    allok = R1 and R2 and R3 and R4 and R5
    print("\nstress gate (integrity red-lines):")
    print(f"  [{'PASS' if R1 else 'FAIL'}] R1 null effect NEVER acted on (no false certification)")
    print(f"  [{'PASS' if R2 else 'FAIL'}] R2 strong effect recovered + acted ({strong_ok}/{strong_total})")
    print(f"  [{'PASS' if R3 else 'FAIL'}] R3 every conclusion replay-verified")
    print(f"  [{'PASS' if R4 else 'FAIL'}] R4 signature recall causally correct")
    print(f"  [{'PASS' if R5 else 'FAIL'}] R5 no crash across the regime sweep")
    print(f"\n  >>> {'PASS — complete-form integrity holds across random regimes' if allok else 'CHECK — red-line risk'}")
    print("\nMeaning: the honesty is not a demo artifact — across varied SCMs the complete form")
    print("never acts on a null effect, recovers and acts on real ones, and every claim is")
    print("recomputable. This is the property a product must never break. Honest: discrete toy SCMs.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
