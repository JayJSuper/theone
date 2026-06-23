"""L0 · energy monitor — watches an evolution's energy trajectory and flags drift
beyond a threshold. It does NOT halt the system; it records the event and raises an
alert (a downstream layer decides whether to abstain). This is a constraint-credential
generator: 'energy stayed within |ΔH| < tol' is a third-party-recomputable claim.
"""
from __future__ import annotations
import numpy as np


class EnergyMonitor:
    def __init__(self, threshold: float = 1e-3) -> None:
        self.threshold = threshold
        self.alerts: list[dict] = []

    def drift(self, energies) -> float:
        e = np.asarray(energies, dtype=float)
        return float(e.max() - e.min())

    def check(self, energies) -> dict:
        """Return a record: drift, whether it exceeded threshold, and append an alert
        if so. Non-fatal (the system keeps running)."""
        d = self.drift(energies)
        rec = {"drift": d, "threshold": self.threshold, "exceeded": d > self.threshold,
               "e0": float(np.asarray(energies)[0])}
        if rec["exceeded"]:
            self.alerts.append(rec)
        return rec


__all__ = ["EnergyMonitor"]
