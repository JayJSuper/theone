"""L1 · LLM adapter — parse an LLM's natural-language causal claim into a STRUCTURED
candidate, with graceful degradation. The adapter is how an LLM enters The One as a
perception organ: it does NOT trust the claim — it converts it into a candidate that
must be VERIFIED by the engine downstream ('LLM proposes, The One verifies').

A well-formed claim yields a candidate {treatment, target, effect, adjustment_set} with
high confidence; a malformed one degrades to confidence < 0.3 (and is not acted upon).
The verification helper compares a claim's stated effect to an engine-computed do() —
corroborating it when they agree and REFUTING it (catching a hallucinated number) when
they do not.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re


@dataclass
class CausalClaim:
    treatment: str | None
    target: str | None
    effect: float | None
    adjustment_set: list[str] = field(default_factory=list)
    confidence: float = 0.0
    raw: str = ""

    def is_actionable(self) -> bool:
        return self.confidence >= 0.3 and self.treatment is not None and self.target is not None


_VAR = r"([A-Za-z_][A-Za-z0-9_]*)"
_NUM = r"([-+]?\d*\.?\d+)"
_PAIR_PATTERNS = [
    rf"effect of {_VAR} on {_VAR}",
    rf"{_VAR}\s*(?:->|→|causes|increases|decreases|affects|drives)\s*{_VAR}",
    rf"do\(\s*{_VAR}\s*\).*?{_VAR}",
]
_EFFECT_PATTERNS = [rf"(?:is|by|=|of)\s*{_NUM}", rf"effect.*?{_NUM}"]


def _find_pair(text: str):
    for pat in _PAIR_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _find_effect(text: str):
    for pat in _EFFECT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1))
                if -1.0 <= v <= 1.0:
                    return v
            except ValueError:
                pass
    return None


def _find_adjustment(text: str):
    m = re.search(r"(?:adjust(?:ing)?|controlling|conditioning|given)\s+(?:for\s+)?(.+?)(?:\.|$)",
                  text, flags=re.IGNORECASE)
    if not m:
        return []
    return [v for v in re.findall(_VAR, m.group(1)) if v.lower() not in ("for", "and")]


class LLMAdapter:
    def parse(self, text: str) -> CausalClaim:
        treatment, target = _find_pair(text)
        if treatment is None or target is None:
            return CausalClaim(None, None, None, confidence=0.1, raw=text)
        effect = _find_effect(text)
        adj = _find_adjustment(text)
        # confidence: structure found is necessary; a parsed effect raises it
        conf = 0.6 + (0.3 if effect is not None else 0.0) + (0.1 if adj else 0.0)
        return CausalClaim(treatment, target, effect, adj, round(min(conf, 1.0), 2), text)

    @staticmethod
    def verify_against_engine(claim: CausalClaim, engine_effect: float, tol: float = 0.05) -> dict:
        """Compare the LLM's stated effect to an engine-computed effect. The LLM is a
        proposer; this is the verification that decides corroborate vs refute."""
        if claim.effect is None:
            return {"verdict": "unverifiable", "reason": "claim states no numeric effect"}
        gap = abs(claim.effect - engine_effect)
        return {"verdict": "corroborated" if gap <= tol else "refuted",
                "claim_effect": claim.effect, "engine_effect": round(engine_effect, 4),
                "gap": round(gap, 4), "tol": tol}


__all__ = ["LLMAdapter", "CausalClaim"]
