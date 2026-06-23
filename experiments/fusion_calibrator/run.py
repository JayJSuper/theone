"""Fusion (10-layer plan · L2) · confidence calibrator — and its honest endpoint.

We reproduce the metacognition finding (self-reported confidence ~flat-high, decoupled
from correctness) and show what calibration can and cannot do:
  • it DROPS the population ECE sharply (good, real),
  • but on a SIGNAL-LESS self-report its honest endpoint is to map every score toward
    the base rate (~0.5 = "I can't tell") — it never recovers WHICH instance is right.
  • only a recomputable credential resolves a specific instance.

So calibration is a useful population diagnostic, strictly subordinate to the per-instance
recomputable credential. This is the fusion done with discipline: adopt the tool, keep
the honest boundary.

Run:  .venv/bin/python experiments/fusion_calibrator/run.py
"""
from __future__ import annotations
import numpy as np

from theone.metacognition import ConfidenceCalibrator, ConfidenceCalibratorConfig, CalibrationMethod


def main():
    print("=== Fusion L2: confidence calibrator + its honest endpoint ===\n")
    rng = np.random.default_rng(0)
    n = 2000
    # anti-calibrated self-reports: confidence ~flat-high, correctness ~ coin flip
    # (the shape we measured across gpt/gemini past the cliff)
    labels = (rng.random(n) < 0.5).astype(float)
    raw_conf = rng.uniform(0.85, 1.0, n)            # ~0.92 mean, no signal about labels
    tr, te = slice(0, 1500), slice(1500, n)

    cal = ConfidenceCalibrator(ConfidenceCalibratorConfig(method=CalibrationMethod.ISOTONIC))
    cal.fit(raw_conf[tr], labels[tr])
    cal_conf = np.array([cal.calibrate(c).calibrated_confidence for c in raw_conf[te]])

    ece_raw = cal.compute_ece(raw_conf[te], labels[te])
    ece_cal = cal.compute_ece(cal_conf, labels[te])
    print(f"population ECE:  raw self-report = {ece_raw:.3f}  ->  calibrated = {ece_cal:.3f}")
    print(f"  calibrated scores collapse toward the base rate: mean={cal_conf.mean():.3f} "
          f"std={cal_conf.std():.3f}  (≈0.5 = 'I can't tell')")

    # the honest endpoint: can EITHER score tell a right answer from a wrong one?
    def discrimination(scores):
        return abs(scores[labels[te] == 1].mean() - scores[labels[te] == 0].mean())
    print(f"\nper-instance discrimination (|mean score on correct − on wrong|):")
    print(f"  raw self-report = {discrimination(raw_conf[te]):.3f}   "
          f"calibrated = {discrimination(cal_conf):.3f}   (both ≈0: neither resolves an instance)")
    print("  → calibration lowers ECE by ADMITTING ignorance (→base rate), not by gaining signal.")
    print("    Only a recomputable credential (independent recompute) resolves a specific answer —")
    print("    which is exactly The One's thesis: trust the recompute, not the self-report.")

    ok = (ece_cal < ece_raw - 0.2 and discrimination(cal_conf) < 0.1)
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}  (ECE drops sharply; discrimination stays ~0)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
