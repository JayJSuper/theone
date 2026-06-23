"""Confidence calibrator (Platt / Isotonic) — fused from the 10-layer plan's L2.

Honest scope (the part the plan under-states): calibration repairs an AGGREGATE
statistic (Expected Calibration Error) over a population. It does NOT make any single
self-reported confidence trustworthy the way a recomputable credential does. Our own
metacognition experiments (registry NOTE-011/028/031) show self-reported LLM confidence
is anti-calibrated per-instance; a calibrator can fix the population ECE yet still leave
an individual "90% confident" answer wrong. So this module is a useful population-level
diagnostic, explicitly subordinate to the per-instance recomputable credential — it
calibrates a score; it does not replace verification.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from theone.core.exceptions import ValidationError


class CalibrationMethod:
    PLATT = "platt"
    ISOTONIC = "isotonic"


@dataclass
class ConfidenceCalibratorConfig:
    method: str = CalibrationMethod.ISOTONIC
    n_bins: int = 10
    min_samples_for_calibration: int = 100


@dataclass
class CalibrationResult:
    calibrated_confidence: float
    original_confidence: float
    ece: Optional[float] = None
    method_used: Optional[str] = None


class ConfidenceCalibrator:
    """Calibrate raw confidence scores against observed correctness (Platt/Isotonic)."""

    def __init__(self, config: Optional[ConfidenceCalibratorConfig] = None) -> None:
        self.config = config or ConfidenceCalibratorConfig()
        if self.config.method not in (CalibrationMethod.PLATT, CalibrationMethod.ISOTONIC):
            raise ValidationError(f"Unsupported method: {self.config.method}")
        self._model = None
        self._is_fitted = False

    def fit(self, confidences: np.ndarray, labels: np.ndarray) -> None:
        confidences = np.asarray(confidences, dtype=float)
        labels = np.asarray(labels, dtype=float)
        if len(confidences) != len(labels):
            raise ValidationError("confidences and labels must have the same length")
        if len(confidences) < self.config.min_samples_for_calibration:
            raise ValidationError(f"Need at least {self.config.min_samples_for_calibration} samples")
        if self.config.method == CalibrationMethod.PLATT:
            self._model = LogisticRegression()
            self._model.fit(confidences.reshape(-1, 1), labels)
        else:
            self._model = IsotonicRegression(out_of_bounds="clip")
            self._model.fit(confidences, labels)
        self._is_fitted = True

    def calibrate(self, confidence: float) -> CalibrationResult:
        if not self._is_fitted:
            raise RuntimeError("Calibrator must be fitted before calibrate()")
        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be in [0, 1], got {confidence}")
        if self.config.method == CalibrationMethod.PLATT:
            cal = float(self._model.predict_proba(np.array([[confidence]]))[0, 1])
        else:
            cal = float(self._model.predict(np.array([confidence]))[0])
        return CalibrationResult(cal, confidence, method_used=self.config.method)

    def calibrate_batch(self, confidences: np.ndarray) -> List[CalibrationResult]:
        return [self.calibrate(float(c)) for c in np.asarray(confidences, dtype=float)]

    def compute_ece(self, confidences: np.ndarray, labels: np.ndarray) -> float:
        """Expected Calibration Error over n_bins equal-width bins."""
        confidences = np.asarray(confidences, dtype=float)
        labels = np.asarray(labels, dtype=float)
        n_bins = self.config.n_bins
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(confidences, edges, right=True), 1, n_bins)
        ece = 0.0
        for b in range(1, n_bins + 1):
            mask = idx == b
            if not np.any(mask):
                continue
            ece += (np.sum(mask) / len(confidences)) * abs(np.mean(labels[mask]) - np.mean(confidences[mask]))
        return float(ece)

    def is_fitted(self) -> bool:
        return self._is_fitted

    def reset(self) -> None:
        self._model = None
        self._is_fitted = False


__all__ = ["ConfidenceCalibrator", "ConfidenceCalibratorConfig",
           "CalibrationResult", "CalibrationMethod"]
