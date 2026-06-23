"""The credential spine — the cross-cutting invariant of The One's fused architecture.

This is the soul of the fusion (`docs/FUSION_ARCHITECTURE.md` §3): every layer
(L0..L5) processes its input and MUST return a `LayerVerdict` that is either
  • ANSWER  — carrying a third-party-RECOMPUTABLE `Credential`, or
  • ABSTAIN — carrying a machine-readable reason (refusing to guess is correct).

It generalizes `os_loop_constrained`'s two orthogonal gates (recomputable AND
admissible) to a 6-layer bus. The decisive property: an ANSWER whose credential
does NOT reproduce within tolerance is automatically DOWNGRADED to ABSTAIN — trust
comes from independent recomputation, never from a layer's confidence. This is the
direct structural answer to the LLM failure mode the project exists to prevent
(confident-narrow-wrong; cf. native_causal_latent probe 5).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
import numpy as np

from theone.core.exceptions import ContractViolationError


class Decision(Enum):
    ANSWER = "answer"
    ABSTAIN = "abstain"


def _numeric_gap(a: Any, b: Any) -> float:
    """Discrepancy between two values. Bools/strings/None compare by equality
    (gap 0 or 1); numeric scalars/arrays compare by max-abs difference."""
    if isinstance(a, bool) or isinstance(b, bool):
        return 0.0 if bool(a) == bool(b) else 1.0
    if isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return 0.0 if a == b else 1.0
    aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        return float("inf")
    return float(np.max(np.abs(aa - bb))) if aa.size else 0.0


@dataclass
class Credential:
    """A layer's machine-checkable receipt. `value` is the claim's quantity; trust
    is established by reproducing it — either in-process via `recompute()` or by a
    third party re-deriving `recompute_digest`. `regime` declares applicability
    limits (e.g. NOTE-004's 'computation-exact, structure-assumed')."""
    layer: str
    claim: str
    value: Any = None
    evidence: dict[str, Any] = field(default_factory=dict)
    regime: str = "unspecified"
    recompute: Optional[Callable[[], Any]] = None
    recompute_digest: Optional[str] = None
    tolerance: float = 1e-9

    def verify(self) -> tuple[bool, dict[str, Any]]:
        """Independently reproduce `value` and check agreement within tolerance.
        Returns (ok, info). With no live recompute, a third party must check the
        digest; we report that honestly rather than claiming verification."""
        if self.recompute is None:
            ok = self.recompute_digest is not None
            return ok, {"mode": "digest-only", "digest": self.recompute_digest}
        try:
            repro = self.recompute()
        except Exception as e:  # fail safe: a recompute that throws is NOT verified
            return False, {"mode": "live-recompute", "error": f"{type(e).__name__}: {e}"}
        gap = _numeric_gap(self.value, repro)
        return gap <= self.tolerance, {"mode": "live-recompute", "gap": gap}


@dataclass
class LayerVerdict:
    decision: Decision
    layer: str
    credential: Optional[Credential] = None
    reason: Optional[str] = None      # required when ABSTAIN
    value: Any = None                 # data passed forward to the next layer

    def is_answer(self) -> bool:
        return self.decision is Decision.ANSWER

    @classmethod
    def answer(cls, layer: str, credential: Credential, value: Any = None) -> "LayerVerdict":
        return cls(Decision.ANSWER, layer, credential=credential, value=value)

    @classmethod
    def abstain(cls, layer: str, reason: str) -> "LayerVerdict":
        return cls(Decision.ABSTAIN, layer, reason=reason)


class CredentialedLayer(ABC):
    """Base class every layer (L0..L5) implements. Subclasses define `process`;
    `run` enforces the spine contract — including the auto-downgrade of any ANSWER
    whose credential fails to recompute."""
    name: str = "unnamed"
    layer_index: int = -1

    @abstractmethod
    def process(self, inputs: Any) -> LayerVerdict:
        ...

    def run(self, inputs: Any) -> LayerVerdict:
        v = self.process(inputs)
        if v.decision is Decision.ANSWER:
            if v.credential is None:
                raise ContractViolationError(f"{self.name}: ANSWER without a credential")
            ok, info = v.credential.verify()
            if not ok:
                # the spine's core guarantee: an unverifiable 'answer' becomes ABSTAIN
                return LayerVerdict.abstain(
                    self.name, f"credential did not recompute ({info})")
        elif not v.reason:
            raise ContractViolationError(f"{self.name}: ABSTAIN without a reason")
        return v


@dataclass
class SystemVerdict:
    decision: Decision
    credentials: list[Credential] = field(default_factory=list)
    abstained_at: Optional[str] = None
    reason: Optional[str] = None

    def is_answer(self) -> bool:
        return self.decision is Decision.ANSWER


class Spine:
    """Runs layers in topological order and short-circuits on the first ABSTAIN
    (the abstain bus). A system ANSWER means every layer answered AND every
    credential independently recomputed — a stacked, end-to-end verifiable receipt."""

    def __init__(self, layers: list[CredentialedLayer]):
        self.layers = sorted(layers, key=lambda L: L.layer_index)

    def run(self, inputs: Any) -> SystemVerdict:
        creds: list[Credential] = []
        state = inputs
        for layer in self.layers:
            v = layer.run(state)
            if v.decision is Decision.ABSTAIN:
                return SystemVerdict(Decision.ABSTAIN, creds,
                                     abstained_at=layer.name, reason=v.reason)
            creds.append(v.credential)
            if v.value is not None:
                state = v.value
        return SystemVerdict(Decision.ANSWER, creds)


__all__ = [
    "Decision", "Credential", "LayerVerdict", "CredentialedLayer",
    "SystemVerdict", "Spine",
]
