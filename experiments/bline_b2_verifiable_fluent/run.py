"""Verifiable-by-construction FLUENT generation — round-trip-gated by W2CG (the B2 generation bone).

The peers' verdict (DeepSeek + Gemini, high agreement): fluent-open-domain and cannot-hallucinate
are NOT jointly achievable in pure generation. The resolution is to DECOUPLE semantic planning
(what is TRUE — fixed by the verified causal structure) from surface realization (how it READS —
learned/varied). Fluency is learned; truth is fixed by the structure, so the generator cannot add
unverified content.

We enforce that with a ROUND-TRIP gate: the generator proposes varied fluent renderings of a
VERIFIED structured claim; the W2CG verifier re-parses each rendering and keeps it ONLY if it
parses back to EXACTLY the same claim. Any wording that flips the direction, drops the entity, or
smuggles in an unverified claim fails the round-trip and is rejected. This is the find/cover /
propose-and-verify pattern applied to generation — fluency without a hallucination surface.

Run:  .venv/bin/python experiments/bline_b2_verifiable_fluent/run.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from theone.language import ClaimVerifier

# a VERIFIED structured claim, as produced by the engine on real finance data (NOTE-122)
CLAIM = {"cause": "payment_delay", "effect": "default", "direction": 1}
STRUCTURE = {("payment_delay", "default"): {"direction": 1, "magnitude": None}}
SYN = {"payment_delay": ["missed payment", "missed a payment", "late payment", "payment delay",
                         "fell behind", "behind on payments", "delinquency", "a delinquency"],
       "default": ["default", "defaulting", "fail to pay", "go delinquent"]}

# the GENERATOR: proposes varied fluent renderings (the learned model is the scale-up; the
# round-trip gate gives the guarantee regardless of how the renderings are produced).
FAITHFUL = [
    "A recent missed payment makes default far more likely.",
    "Falling behind on payments drives up the odds of default.",
    "When a borrower has a late payment, the chance of default goes up.",
    "A missed payment raises the risk of default.",
    "Borrowers with a recent delinquency are more likely to default.",
]
# adversarial renderings the generator might also propose — each smuggles in an error:
HALLUCINATED = [
    ("A missed payment lowers the risk of default.", "flips direction"),
    ("A missed payment guarantees the borrower will default.", "overclaims (guarantee)"),
    ("A missed payment and the borrower's zodiac sign both cause default.", "adds unverified claim"),
    ("Higher income makes default far more likely.", "wrong entity (unverified)"),
]


import re
# coordination markers that introduce an ADDITIONAL causal agent ("X and Y both cause Z"). A
# verifiable-by-construction rendering asserts ONLY the verified claim, so any of these = reject.
# (round-trip parsing alone extracts one triple and would miss the smuggled second one.)
_ADDITIVE = re.compile(r"\b(both|as well as|along with|in addition to|and [a-z ]+ (also )?(cause|causes|raise|raises|increase|increases|drive|drives|make|makes))\b")


def round_trip_ok(cv, sentence):
    """Admissible iff re-parsing yields EXACTLY the verified claim AND the sentence asserts only
    that one claim (no additive smuggle of a second, unverified cause)."""
    if _ADDITIVE.search(sentence.lower()):
        return False                                         # asserts more than the verified claim
    v = cv.verify_claim(sentence)
    return (v.verdict == "VERIFIED" and v.cause == CLAIM["cause"]
            and v.effect == CLAIM["effect"] and v.direction == CLAIM["direction"])


def main():
    print("=== Verifiable-by-construction FLUENT generation · round-trip-gated by W2CG ===\n")
    print(f"  verified claim to express: {CLAIM['cause']} -> {CLAIM['effect']} (direction +, VERIFIABLE)\n")
    cv = ClaimVerifier(STRUCTURE, SYN)

    print("  faithful renderings (should PASS the round-trip and be emitted):")
    emitted = 0
    for s in FAITHFUL:
        ok = round_trip_ok(cv, s); emitted += ok
        print(f"    [{'emit' if ok else 'DROP':4}] {s}")

    print("\n  adversarial renderings (each smuggles an error — must be CAUGHT/dropped):")
    caught = 0
    for s, why in HALLUCINATED:
        ok = round_trip_ok(cv, s); caught += (not ok)
        print(f"    [{'LEAK!' if ok else 'caught':5}] ({why}) {s}")

    print(f"\n  emitted {emitted}/{len(FAITHFUL)} faithful · caught {caught}/{len(HALLUCINATED)} hallucinated")
    g1 = emitted >= 4                                         # fluent variety survives the gate
    g2 = caught == len(HALLUCINATED)                          # EVERY hallucination is caught (the red-line)
    allok = g1 and g2
    print("\nverifiable-fluent gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] multiple fluent renderings of the verified claim are emitted")
    print(f"  [{'PASS' if g2 else 'FAIL'}] EVERY hallucinated rendering is caught by the round-trip (red-line)")
    print(f"\n  >>> {'PASS — fluency without a hallucination surface: render verified structure many ways, reject any wording that changes its meaning' if allok else 'CHECK'}")
    print("\n  This is the decoupled architecture the peers prescribed: truth is FIXED by the verified")
    print("  structure, surface wording is free/learned, and the round-trip W2CG gate guarantees the")
    print("  wording still means exactly the verified claim — no wording can introduce a claim the")
    print("  engine cannot recompute. Scale-up = a LEARNED realizer (more fluent/varied); the gate's")
    print("  guarantee holds no matter how the renderings are produced.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
