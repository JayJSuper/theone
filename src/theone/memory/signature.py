"""Causal signature — pillar 2's first-class citizen.

The thesis of the memory pillar: a memory relevant to a *causal* decision should be
indexed by its **de-confounded causal fingerprint**, not its surface text. Two
memories can read almost identically yet carry opposite true effects once confounding
is adjusted away; surface (embedding) retrieval cannot tell them apart, signature
retrieval can. The signature is *derived from a computation-pillar credential* — so the
memory pillar's confounding-immunity is **inherited**, not re-earned (the cross-pillar
unity claim, made concrete).

A CausalSignature is the tuple that the credential already certifies:
  (treatment → target, adjustment set, de-confounded effect, regime).
Retrieval matches on this tuple, with the effect compared numerically and the
structure compared exactly — so a confounded look-alike with a different true effect
lands far away in signature space even when it is a near-neighbour in text space.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class CausalSignature:
    treatment: str
    target: str
    adjustment_set: tuple            # variables conditioned on (order-insensitive)
    effect: float                    # de-confounded P(target=1 | do(treatment=1))
    regime: str = "default"          # e.g. "normal" | "stressed"

    def __post_init__(self):
        # normalise the adjustment set so equal sets compare equal
        object.__setattr__(self, "adjustment_set", tuple(sorted(self.adjustment_set)))

    # --- the stable structural key (exact-match retrieval) -------------------
    def structure_key(self) -> str:
        adj = ",".join(self.adjustment_set)
        return f"{self.treatment}->{self.target}|adj=[{adj}]|regime={self.regime}"

    # --- derive from a computation-pillar credential -------------------------
    @classmethod
    def from_credential(cls, cred: dict) -> "CausalSignature":
        """A credential (the computation pillar's output) already carries exactly the
        fields a causal signature needs. This is the inheritance: the memory's
        de-confounding is the credential's de-confounding, not a new estimate."""
        return cls(
            treatment=cred["treatment"],
            target=cred["target"],
            adjustment_set=tuple(cred.get("adjustment_set", ())),
            effect=float(cred["effect"]),
            regime=cred.get("regime", "default"),
        )

    # --- signature-space distance -------------------------------------------
    def distance(self, other: "CausalSignature", effect_weight: float = 1.0) -> float:
        """Distance in *causal* space. Same structure (treatment/target/adjustment/
        regime) and same de-confounded effect → 0. A surface look-alike with a
        different true effect, or a different adjustment set, is far. Structural
        mismatch dominates (it means the two memories are about different causal
        questions); effect gap is the within-structure refinement."""
        struct_mismatch = 0.0 if self.structure_key() == other.structure_key() else 1.0
        effect_gap = abs(self.effect - other.effect)
        return struct_mismatch + effect_weight * effect_gap

    # --- deterministic vector (for signature-space ANN if desired) ----------
    def vector(self, dim: int = 64) -> np.ndarray:
        """Deterministic embedding of the structural key, scaled by the effect.
        Distinct structures map to near-orthogonal directions; the effect modulates
        magnitude so same-structure/different-effect memories separate too."""
        h = hashlib.sha256(self.structure_key().encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        v = rng.standard_normal(dim)
        v = v / (np.linalg.norm(v) + 1e-9)
        return v * (1.0 + self.effect)

    def to_dict(self) -> dict:
        return {"treatment": self.treatment, "target": self.target,
                "adjustment_set": list(self.adjustment_set),
                "effect": self.effect, "regime": self.regime}

    @classmethod
    def from_dict(cls, d: dict) -> "CausalSignature":
        return cls(treatment=d["treatment"], target=d["target"],
                   adjustment_set=tuple(d["adjustment_set"]),
                   effect=float(d["effect"]), regime=d.get("regime", "default"))
