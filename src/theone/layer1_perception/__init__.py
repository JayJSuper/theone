"""Layer 1 · continuous signal perception — the system's sensory entry.

No tokenizer: signals enter as continuous streams. The SSM encoder maps them to a
stable latent trajectory (spectral radius < 1) reconstructable by a linear decoder;
the temporal lock enforces a strict nanosecond order; the modality registry routes
sensors/LLMs. PerceptionLayer wraps the encoder onto the spine and emits the latent
as the L1->L2 connector.
"""
from theone.layer1_perception.ssm_encoder import SSMEncoder
from theone.layer1_perception.temporal_locking import TemporalLock, TemporalConflictError
from theone.layer1_perception.modality_registry import (
    ModalityRegistry, ModalityConfig, UnknownModalityError,
)
from theone.layer1_perception.perception_layer import PerceptionLayer
from theone.layer1_perception.llm_adapter import LLMAdapter, CausalClaim

__all__ = [
    "SSMEncoder", "TemporalLock", "TemporalConflictError",
    "ModalityRegistry", "ModalityConfig", "UnknownModalityError",
    "PerceptionLayer", "LLMAdapter", "CausalClaim",
]
