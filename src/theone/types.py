"""Core types. CC-02. All probability values are validated at construction (frozen ruling:
validation replaces normalization - out-of-range values are REJECTED, never rescaled)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


class TheOneError(Exception):
    """Base error."""


class CalibrationRequiredError(TheOneError):
    """Raised when a statistical judgment is requested before delta_min calibration.
    Frozen rule: A7 must not run with an uncalibrated substantive threshold."""


class GraphValidationError(TheOneError):
    """Raised on invalid graph/CPT construction (cycles, bad probabilities)."""


@dataclass(frozen=True)
class Variable:
    name: str
    states: tuple = (0, 1)


@dataclass
class QueryResult:
    value: float
    method: str
    details: dict = field(default_factory=dict)


@dataclass
class CompareResult:
    obs_ate: float
    int_ate: float
    are_different: Any  # True / False / "statistically_significant_below_substantive_threshold"
    stats: dict = field(default_factory=dict)


@dataclass
class TheOneConfig:
    seed: int = 42
    bootstrap_B: int = 1000
    alpha: float = 0.05
    delta_min: float | None = None  # None = uncalibrated; A7 must refuse to run (frozen)
    numeric_tol: float = 1e-9
