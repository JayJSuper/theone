"""VerifiedReporter — generate fluent natural-language from VERIFIED engine findings, round-trip-
gated so no emitted sentence can assert anything the engine did not certify (NOTE-126/127).

The generation counterpart to ClaimVerifier: where ClaimVerifier judges incoming claims, the
reporter EMITS claims — but only renderings that re-parse to exactly the verified finding and
smuggle in no extra causal assertion are emitted; everything else is held back. Honest zones
(VERIFIABLE / UNCERTAINTY_QUANTIFIED / REJECT) are surfaced verbatim, so the wording can never
claim more confidence than the engine certified. This is fluency without a hallucination surface.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
from theone.language.claim_verifier import ClaimVerifier

# coordination that introduces a SECOND causal agent — a verifiable rendering asserts only its one finding
_ADDITIVE = re.compile(r"\b(both|as well as|along with|in addition to|"
                       r"and [a-z ]+ (also )?(cause|causes|raise|raises|increase|increases|make|makes))\b")


@dataclass
class Finding:
    cause: str
    effect: str
    direction: int                 # +1 / -1
    zone: str                      # VERIFIABLE / UNCERTAINTY_QUANTIFIED / REJECT
    ate: Optional[float] = None
    e_value: Optional[float] = None


class VerifiedReporter:
    """label: canonical-node -> human phrase (e.g. {'payment_delay': 'a recent missed payment'}).
    entity_syn: canonical-node -> surface synonyms (for the round-trip parse)."""

    def __init__(self, label: dict, entity_syn: dict) -> None:
        self.label = label
        self.entity_syn = entity_syn

    def _render(self, f: Finding) -> str:
        subj = self.label.get(f.cause, f.cause); obj = self.label.get(f.effect, f.effect)
        verb = "raises" if f.direction >= 0 else "lowers"
        eff = f" (effect {f.ate:+.2f})" if f.ate is not None else ""
        if f.zone == "VERIFIABLE":
            ev = f", and robust to unobserved confounding up to a risk-ratio of {f.e_value:.1f}" if f.e_value else ""
            return f"{subj} {verb} the risk of {obj}{eff}; this is verified{ev}.".capitalize()
        if f.zone == "UNCERTAINTY_QUANTIFIED":
            ev = f"a confounder of risk-ratio {f.e_value:.1f} could explain it away" if f.e_value else "it is fragile"
            return f"There is uncertainty-quantified evidence that {subj} {verb} {obj}{eff}, but {ev}."
        return f"The engine could not certify a causal effect of {subj} on {obj} — reported as inconclusive."

    def emit(self, f: Finding):
        """Return (sentence, verifiable). For directional (non-REJECT) findings the sentence must
        round-trip to exactly this finding with no additive smuggle, else it is held back."""
        s = self._render(f)
        if f.zone == "REJECT":
            return s, True                                   # honest inconclusive — asserts no causal claim
        if _ADDITIVE.search(s.lower()):
            return s, False
        struct = {(f.cause, f.effect): {"direction": f.direction, "magnitude": None}}
        v = ClaimVerifier(struct, self.entity_syn).verify_claim(s)
        ok = (v.verdict == "VERIFIED" and v.cause == f.cause and v.effect == f.effect
              and v.direction == f.direction)
        return s, ok

    def report(self, findings) -> dict:
        """Render every finding; emit only the verifiable-by-construction ones (REJECT findings are
        emitted as honest 'inconclusive'). Returns the report lines + a guarantee flag."""
        lines, held = [], []
        for f in findings:
            s, ok = self.emit(f)
            (lines if ok else held).append(s)
        return {"report": lines, "held_back": held,
                "verifiable_by_construction": True, "n_emitted": len(lines), "n_held": len(held)}


__all__ = ["VerifiedReporter", "Finding"]
