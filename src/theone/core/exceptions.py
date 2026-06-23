"""Global exceptions for the fused 6-layer architecture.

Reconciled with the existing verified codebase: the base `TheOneError` and
`GraphValidationError` already live in `theone.types` (frozen). We re-export them
here (single source of truth) and extend the hierarchy with the layer-specific
errors the blueprint's L0/L2 introduce. Abstention is deliberately NOT an
exception — it is a normal `LayerVerdict` (a layer refusing to guess is correct
behavior, not a failure). See `theone.core.spine`.
"""
from __future__ import annotations

# single source of truth — do not redefine, re-export the frozen base
from theone.types import TheOneError, GraphValidationError


class PhysicsViolationError(TheOneError):
    """Raised when a state violates a physical constraint (L0)."""


class EnergyConservationError(PhysicsViolationError):
    """Raised when energy drift exceeds the threshold (L0 energy monitor)."""


class CausalAcyclicityError(TheOneError):
    """Raised when a graph that must be a DAG contains a cycle (L2)."""


class ContractViolationError(TheOneError):
    """Raised when a layer breaks the spine contract (e.g. ANSWER without a
    recomputable credential, or ABSTAIN without a reason)."""


class ValidationError(TheOneError):
    """Invalid input/argument to a module (general-purpose; vocabulary adopted from
    the 10-layer product plan's module specs)."""


__all__ = [
    "TheOneError", "GraphValidationError",
    "PhysicsViolationError", "EnergyConservationError",
    "CausalAcyclicityError", "ContractViolationError", "ValidationError",
]
