"""The One · core — the fusion seam shared by all 6 layers.

This package is the load-bearing foundation of the fused architecture
(`docs/FUSION_ARCHITECTURE.md`): shared data contracts + the credential spine
that threads every layer (L0..L5) with one verifiable invariant — each layer
emits a third-party-recomputable credential or ABSTAINS, never a confident guess.
"""
from theone.core.contracts import StateVector, Observation, Graph, Action
from theone.core.spine import (
    Decision, Credential, LayerVerdict, CredentialedLayer, Spine,
)
from theone.core import exceptions

__all__ = [
    "StateVector", "Observation", "Graph", "Action",
    "Decision", "Credential", "LayerVerdict", "CredentialedLayer", "Spine",
    "exceptions",
]
