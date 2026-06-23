"""Shared data contracts for the fused 6-layer architecture (blueprint §4).

These are the data-plane carriers that flow between layers. The control-plane
verdict/credential lives in `theone.core.spine`. Kept dependency-light (numpy
only) so Layer 0 can import them without pulling in torch/networkx.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import numpy as np

Modality = Literal["optical", "acoustic", "force", "em", "gravity", "text"]
ActionType = Literal["file_write", "shell_cmd", "api_call"]


@dataclass
class StateVector:
    """System state at an instant — generalized coordinates/momenta + energy.
    The (q, p) split is what lets Layer 0 check symplectic/energy invariants;
    a layer that does not use Hamiltonian structure may leave momentum zero."""
    coordinates: np.ndarray            # q, shape (latent_dim,)
    momentum: np.ndarray               # p, shape (latent_dim,)
    energy: float
    timestamp_ns: int

    @property
    def latent_dim(self) -> int:
        return int(self.coordinates.shape[0])


@dataclass
class Observation:
    """One external perception (sensor stream or parsed LLM output)."""
    source_id: str
    modality: Modality
    data: np.ndarray
    timestamp_ns: int
    confidence: float = 1.0            # LLMAdapter degrades this on bad parse


@dataclass
class Graph:
    """Directed weighted graph; can assert the DAG constraint Layer 2 needs."""
    nodes: list[str]
    edges: list[tuple[str, str, float]]   # (src, dst, weight)

    def is_dag(self) -> bool:
        """Kahn's algorithm — dependency-free acyclicity check."""
        indeg = {n: 0 for n in self.nodes}
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for s, d, _ in self.edges:
            if s not in indeg or d not in indeg:
                return False
            indeg[d] += 1
            adj[s].append(d)
        queue = [n for n, k in indeg.items() if k == 0]
        seen = 0
        while queue:
            n = queue.pop()
            seen += 1
            for m in adj[n]:
                indeg[m] -= 1
                if indeg[m] == 0:
                    queue.append(m)
        return seen == len(self.nodes)


@dataclass
class Action:
    """A world-affecting action — must carry a credential before it executes (L5)."""
    id: str
    type: ActionType
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = True
    credential: Optional[Any] = None   # theone.core.spine.Credential when issued


__all__ = ["Modality", "ActionType", "StateVector", "Observation", "Graph", "Action"]
