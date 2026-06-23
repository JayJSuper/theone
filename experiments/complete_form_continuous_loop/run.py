"""Complete-form CONTINUOUS cognitive loop — analyze_continuous -> remember -> recall -> act.

Aligns the continuous path with the binary cognitive OS: a continuous-outcome causal
conclusion (TARNet ATE + reproducible-replay + continuous three-zone) is REMEMBERED by its
de-confounded signature, RECALLED for a decision, and turned into a credentialed ACTION — or
an honest abstain. Binary and continuous now share one cognitive-OS surface.

Run:  .venv/bin/python experiments/complete_form_continuous_loop/run.py
"""
from __future__ import annotations
import warnings
import numpy as np

warnings.filterwarnings("ignore")
from theone.native import CompleteForm
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature


def make(n, ate, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * X[:, 0] - 0.5 * X[:, 1])))).astype(np.float32)
    y = (2.0 + X[:, 0] + 0.5 * X[:, 2] + ate * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, t, y


def main():
    print("=== complete-form CONTINUOUS cognitive loop · analyze -> remember -> recall -> act ===\n")
    cf = CompleteForm()
    mem = SovereignMemory(":memory:")

    # strong continuous effect -> verify -> remember -> recall -> act
    Xs, ts, ys = make(1600, 3.0, 1)
    rS = cf.analyze_continuous(Xs, ts, ys)
    midS = mem.remember("continuous treatment effect (strong)", rS.credential, source="cf-cont")
    actS = cf.recommend(rS, min_effect=0.2)
    qS = CausalSignature(treatment="T", target="Y_continuous",
                         adjustment_set=("<all covariates>",), effect=rS.effect,
                         regime=rS.credential["regime"])
    recalledS = mem.recall_for_decision(qS, k=1)[0]
    print(f"STRONG: effect={rS.effect:+.3f} zone={rS.zone} -> remembered(id {midS}) "
          f"-> recalled(id {recalledS.mem_id})")
    print(f"  ACT: {actS['action'].upper()} — {actS.get('decision','')}")

    # near-null continuous effect -> abstain (no action)
    Xn, tn, yn = make(1600, 0.02, 2)
    rN = cf.analyze_continuous(Xn, tn, yn)
    actN = cf.recommend(rN, min_effect=0.2)
    print(f"\nNULL: effect={rN.effect:+.3f} zone={rN.zone}")
    print(f"  ACT: {actN['action'].upper()} — {actN.get('reason','')[:70]}")

    g1 = abs(rS.effect - 3.0) < 0.6 and actS["action"] == "recommend"   # strong recovered + acted
    g2 = recalledS.mem_id == midS                                       # continuous belief recalled
    g3 = actN["action"] == "abstain"                                    # null continuous -> abstain
    g4 = rS.credential["replay_ok"]                                     # replay-verified
    allok = g1 and g2 and g3 and g4
    print("\ncontinuous-loop gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] strong continuous effect recovered + acted")
    print(f"  [{'PASS' if g2 else 'FAIL'}] continuous belief remembered + recalled by signature")
    print(f"  [{'PASS' if g3 else 'FAIL'}] near-null continuous effect -> abstain")
    print(f"  [{'PASS' if g4 else 'FAIL'}] conclusion replay-verified")
    print(f"\n  >>> {'PASS — continuous path shares the full cognitive-OS loop (remember+recall+act)' if allok else 'CHECK'}")
    print("\nMeaning: binary and continuous causal conclusions now share ONE cognitive-OS surface —")
    print("both are verified, remembered by de-confounded signature, recalled, and acted on (or")
    print("abstained) with a credential. Honest: synthetic continuous SCM.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
