"""Unit tests for the fused L2 confidence calibrator."""
from __future__ import annotations
import numpy as np
import pytest

from theone.metacognition import ConfidenceCalibrator, ConfidenceCalibratorConfig, CalibrationMethod
from theone.core.exceptions import ValidationError


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.5).astype(float)
    conf = rng.uniform(0.85, 1.0, n)           # anti-calibrated: flat-high, no signal
    return conf, labels


def test_isotonic_calibration_drops_ece():
    conf, labels = _data()
    cal = ConfidenceCalibrator(ConfidenceCalibratorConfig(method=CalibrationMethod.ISOTONIC))
    cal.fit(conf, labels)
    out = np.array([cal.calibrate(c).calibrated_confidence for c in conf])
    assert cal.compute_ece(out, labels) < cal.compute_ece(conf, labels) - 0.15


def test_platt_method_runs_and_bounds():
    conf, labels = _data(seed=1)
    cal = ConfidenceCalibrator(ConfidenceCalibratorConfig(method=CalibrationMethod.PLATT))
    cal.fit(conf, labels)
    r = cal.calibrate(0.9)
    assert 0.0 <= r.calibrated_confidence <= 1.0 and r.method_used == "platt"


def test_calibrate_before_fit_raises():
    with pytest.raises(RuntimeError):
        ConfidenceCalibrator().calibrate(0.5)


def test_invalid_inputs_raise():
    with pytest.raises(ValidationError):
        ConfidenceCalibrator(ConfidenceCalibratorConfig(method="bogus"))
    cal = ConfidenceCalibrator()
    with pytest.raises(ValidationError):
        cal.fit(np.array([0.9, 0.8]), np.array([1.0]))            # length mismatch
    cal.fit(*_data())
    with pytest.raises(ValidationError):
        cal.calibrate(1.5)                                         # out of [0,1]
