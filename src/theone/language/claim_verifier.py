"""ClaimVerifier — verify a natural-language causal claim against known causal structure.

Parses an English causal sentence into a structured claim (cause, effect, direction, magnitude,
overclaim) via an entity-synonym normalizer + idiom direction lexicon, then judges it against a
caller-supplied causal knowledge base. The non-negotiable red-line: NEVER return VERIFIED for a
claim that contradicts or is absent from the knowledge — abstain (UNVERIFIABLE) instead. This is
the rule-based bridge (NOTE-113/114); a learned extractor (NOTE-115) is the generalization path.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

# default lexicons — callers can extend ENTITY for their domain
INC = ["increase", "increases", "raise", "raises", "boost", "boosts", "cause", "causes", "bump",
       "bumps", "skyrocket", "skyrockets", "improve", "improves", "speed", "speeds", "elevate",
       "elevates", "promote", "promotes", "worsen", "worsens", "lead to", "leads to",
       # common everyday/risk phrasings (odds/likelihood up)
       "more likely", "drives up", "drive up", "drove up", "raises the odds", "raise the odds",
       "raises the risk", "raises the chance", "ups the", "higher risk", "higher chance",
       "increases the odds", "increases the chance", "far more likely", "much more likely",
       "default more", "defaults more", "goes up", "go up", "going up", "shoots up"]
DEC = ["decrease", "decreases", "reduce", "reduces", "lower", "lowers", "cut", "cuts", "prevent",
       "prevents", "protect", "protects", "nudge", "nudges", "inhibit", "inhibits", "alleviate",
       "alleviates", "slow", "slows",
       # common everyday/risk phrasings (odds/likelihood down)
       "less likely", "lowers the odds", "lower the odds", "reduces the odds", "reduces the risk",
       "lower risk", "lower chance", "default less", "defaults less", "tend to default less"]
NULL_PH = ["no effect", "didn't do squat", "did not do squat", "zero effect", "didn't change",
           "did not change", "no impact", "doesn't change", "does not affect", "not affect", "unrelated"]
OVERCLAIM = ["cure", "cures", "guarantee", "guarantees", "miracle", "every time", "no matter what",
             "completely", "everyone who takes", "always"]
STRONG = ["strongly", "dramatically", "massively", "skyrocket", "skyrockets", "surefire", "huge", "greatly"]
SLIGHT = ["slightly", "barely", "marginally", "not by a huge", "a bit"]

VERDICTS = ("VERIFIED", "CONTRADICTED", "UNVERIFIABLE")


@dataclass
class Verdict:
    verdict: str                       # VERIFIED / CONTRADICTED / UNVERIFIABLE
    cause: Optional[str]
    effect: Optional[str]
    direction: Optional[int]           # +1 / -1 / 0 / None
    reason: str


class ClaimVerifier:
    """structure: dict {(cause, effect): {'direction': +1/-1/0, 'magnitude': 0-2 or None}}.
    entity_syn: dict {canonical_node: [surface synonyms/idioms]}."""

    def __init__(self, structure: dict, entity_syn: dict) -> None:
        self.structure = structure
        self.entity_syn = entity_syn
        self.causes = {c for c, _ in structure}
        self.effects = {e for _, e in structure}

    def _entities(self, s: str) -> dict:
        found = {}
        for canon, syns in self.entity_syn.items():
            for w in syns:
                if w in s:
                    found[canon] = s.index(w); break
        return found

    def parse(self, sentence: str) -> Optional[dict]:
        s = " " + sentence.lower().strip().rstrip(".") + " "
        # word-boundary match: substring matching falsely fires ("cure" inside "uncured/secure")
        overclaim = any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in OVERCLAIM)
        null = any(w in s for w in NULL_PH)
        # catch contraction negation ("doesn't/don't/won't prevent" must not read as a clean −);
        # \bn't\b fails inside "doesn't" (no boundary before n), so match the n't suffix directly
        negated = bool(re.search(r"\b(no|not|never|without)\b|n['’]t\b", s)) and not null
        direction = 0 if null else None
        if direction is None:
            if any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in INC): direction = 1
            if any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in DEC): direction = -1
        if "live longer" in s: direction = -1               # idiom: live longer -> less mortality
        if direction is None and not overclaim:
            return None
        if negated and direction not in (0, None):
            direction = 0
        mag = 2 if any(w in s for w in STRONG) else (0 if any(w in s for w in SLIGHT) else None)
        ents = self._entities(s)
        c = sorted([(p, e) for e, p in ents.items() if e in self.causes])
        f = sorted([(p, e) for e, p in ents.items() if e in self.effects])
        return {"cause": c[0][1] if c else None, "effect": f[0][1] if f else None,
                "direction": direction, "magnitude": mag, "overclaim": overclaim}

    def verify_claim(self, sentence: str) -> Verdict:
        claim = self.parse(sentence)
        if claim is None:
            return Verdict("UNVERIFIABLE", None, None, None, "no causal claim parsed")
        key = (claim["cause"], claim["effect"])
        if claim["cause"] is None or claim["effect"] is None or key not in self.structure:
            return Verdict("UNVERIFIABLE", claim["cause"], claim["effect"], claim["direction"],
                           "claim is about something outside verified knowledge — abstaining")
        fact = self.structure[key]
        if claim["overclaim"]:
            return Verdict("CONTRADICTED", *key, claim["direction"], "overclaims a known relationship")
        if claim["direction"] is not None and claim["direction"] != fact["direction"]:
            return Verdict("CONTRADICTED", *key, claim["direction"],
                           f"asserts direction {claim['direction']} but verified is {fact['direction']}")
        if claim["magnitude"] is not None and fact.get("magnitude") is not None \
                and claim["magnitude"] != fact["magnitude"]:
            return Verdict("CONTRADICTED", *key, claim["direction"], "asserts the wrong magnitude")
        return Verdict("VERIFIED", *key, claim["direction"], "matches verified causal knowledge")


__all__ = ["ClaimVerifier", "Verdict"]
