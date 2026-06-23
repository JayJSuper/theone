"""L1 · temporal locking — every observation carries a strictly monotonic nanosecond
timestamp. Provenance and the auditable belief history (Layer 4) depend on a total
temporal order; a duplicate or out-of-order stamp is REJECTED, not silently accepted.
"""
from __future__ import annotations
import time


class TemporalConflictError(Exception):
    """Raised when an incoming timestamp is not strictly greater than the last."""


class TemporalLock:
    def __init__(self) -> None:
        self._last_ns: int = -1
        self._conflicts: int = 0

    def stamp(self) -> int:
        """Issue a fresh, strictly-increasing nanosecond timestamp."""
        ns = time.monotonic_ns()
        if ns <= self._last_ns:        # clock didn't advance enough → force monotonicity
            ns = self._last_ns + 1
        self._last_ns = ns
        return ns

    def accept(self, ns: int) -> int:
        """Accept an externally-provided timestamp iff it is strictly increasing.
        Returns the timestamp; raises TemporalConflictError on conflict."""
        if ns <= self._last_ns:
            self._conflicts += 1
            raise TemporalConflictError(
                f"timestamp {ns} <= last {self._last_ns} (conflict #{self._conflicts})")
        self._last_ns = ns
        return ns

    @property
    def conflicts(self) -> int:
        return self._conflicts

    @property
    def last(self) -> int:
        return self._last_ns


__all__ = ["TemporalLock", "TemporalConflictError"]
