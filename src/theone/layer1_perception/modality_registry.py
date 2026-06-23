"""L1 · modality registry — dynamic registration of perception modalities (optical,
acoustic, force, em, gravity, text...), each with its own config. Known modalities
resolve in O(1); an unknown modality raises rather than guessing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


class UnknownModalityError(KeyError):
    """Raised when querying a modality that is not registered."""


@dataclass
class ModalityConfig:
    name: str
    input_dim: int
    sample_rate_hz: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


class ModalityRegistry:
    def __init__(self) -> None:
        self._reg: dict[str, ModalityConfig] = {}

    def register(self, cfg: ModalityConfig) -> None:
        if cfg.name in self._reg:
            raise ValueError(f"modality {cfg.name!r} already registered")
        self._reg[cfg.name] = cfg

    def unregister(self, name: str) -> None:
        if name not in self._reg:
            raise UnknownModalityError(name)
        del self._reg[name]

    def get(self, name: str) -> ModalityConfig:
        try:
            return self._reg[name]
        except KeyError:
            raise UnknownModalityError(name) from None

    def is_registered(self, name: str) -> bool:
        return name in self._reg

    @property
    def names(self) -> list[str]:
        return list(self._reg)


__all__ = ["ModalityRegistry", "ModalityConfig", "UnknownModalityError"]
