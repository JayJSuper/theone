"""Layer 4 · auditable memory — SovereignMemory re-homed onto the spine.

Wraps the verified signature-indexed, versioned, sovereign store. Retrieval is by
causal signature (not surface text); the credential's recompute re-reads the memory
from persistent storage and re-derives its structure key, so a recall is trusted only
if it is consistent with what is actually persisted.
"""
from theone.layer4_memory.memory_layer import MemoryLayer
from theone.layer4_memory.pattern_recognition import PatternRecognizer
from theone.layer4_memory.conflict_arbitrator import ConflictArbitrator

__all__ = ["MemoryLayer", "PatternRecognizer", "ConflictArbitrator"]
