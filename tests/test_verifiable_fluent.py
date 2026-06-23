"""Regression lock for the B2 generation kill-gate (NOTE-126): verifiable-by-construction fluent
generation. The round-trip gate (re-parse a rendering and require it to map back to EXACTLY the
verified claim, with no additive smuggle) must emit faithful renderings and catch every class of
hallucination. This protects the guarantee even as the language lexicons evolve."""
from __future__ import annotations
import re
from theone.language import ClaimVerifier

CLAIM = {"cause": "payment_delay", "effect": "default", "direction": 1}
STRUCTURE = {("payment_delay", "default"): {"direction": 1, "magnitude": None}}
SYN = {"payment_delay": ["missed payment", "missed a payment", "late payment", "fell behind",
                         "behind on payments", "delinquency", "a delinquency"],
       "default": ["default", "defaulting", "fail to pay"]}
_ADDITIVE = re.compile(r"\b(both|as well as|along with|in addition to|"
                       r"and [a-z ]+ (also )?(cause|causes|raise|raises|increase|increases|make|makes))\b")


def _round_trip_ok(cv, s):
    if _ADDITIVE.search(s.lower()):
        return False
    v = cv.verify_claim(s)
    return (v.verdict == "VERIFIED" and v.cause == CLAIM["cause"]
            and v.effect == CLAIM["effect"] and v.direction == CLAIM["direction"])


def test_faithful_renderings_emit():
    cv = ClaimVerifier(STRUCTURE, SYN)
    faithful = [
        "A recent missed payment makes default far more likely.",
        "Falling behind on payments drives up the odds of default.",
        "A missed payment raises the risk of default.",
        "Borrowers with a recent delinquency are more likely to default.",
    ]
    assert all(_round_trip_ok(cv, s) for s in faithful)


def test_every_hallucination_class_is_caught():
    cv = ClaimVerifier(STRUCTURE, SYN)
    hallucinated = [
        "A missed payment lowers the risk of default.",                       # flip direction
        "A missed payment guarantees the borrower will default.",             # overclaim
        "A missed payment and the borrower's zodiac sign both cause default.", # additive smuggle
        "Higher income makes default far more likely.",                       # wrong/unverified entity
    ]
    assert not any(_round_trip_ok(cv, s) for s in hallucinated)
