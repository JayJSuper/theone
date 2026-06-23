"""Layer 5 · verifiable execution — SafeExecutor re-homed onto the spine.

The SafeExecutor already embodies the spine's philosophy (two orthogonal gates:
sandbox/denylist admissibility AND a recomputable-AND-admissible causal gate, with an
EXECUTE/BLOCK/ABSTAIN decision). This layer maps that decision onto LayerVerdict:
EXECUTE → ANSWER (with a deterministically recomputable decision credential),
BLOCK/ABSTAIN → ABSTAIN.
"""
from theone.layer5_execution.execution_layer import ExecutionLayer

__all__ = ["ExecutionLayer"]
