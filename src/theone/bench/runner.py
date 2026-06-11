"""MVP-2A two-phase runner. CC-09.

Phase 'calibrate' (exploratory, burn-after-use):
  - maps the |tau| distribution of the pure-association estimator on a calibration
    SCM set; freezes delta_min = max(median, 0.05) (frozen scheme, gatekeeper-signed);
  - writes burned_list.json: calibration spec fingerprints are PERMANENTLY excluded
    from the frozen phase (two-stage general clause: calibration sets are burned).
  - NOTE (honest scoping): min/product aggregation calibration for the soft-graph
    sufficiency score lands with the T3 brick; it is intentionally absent here,
    not mocked.

Phase 'frozen' (confirmatory):
  - loads a frozen config (delta_min + grid), REFUSES any spec whose fingerprint
    appears in burned_list.json, runs method-vs-baseline, reports EG distribution
    + RMSE/MAE + conjunctive verdict (Amendments 1/1a/2/2a).
  - v0.1 'method' = oracle engine on the true SCM (pipeline plumbing); the real
    MVP-2A method configuration ships with task batch 02 after Q-C7 freezes the
    grid values.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path
import numpy as np
from ..types import TheOneConfig
from .eg import (SCMGenerator, SCMSpec, abs_errors, eg_score, conjunctive_verdict)


def _assoc_int_estimate(scm) -> float:
    """Pure-association estimator of the interventional ATE: the observational
    OLS slope of Y on X - exactly what a correlational model would output."""
    d = scm.sample()
    x, y = d["X"], d["Y"]
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def run_calibration(grid_config: dict, out_dir: str, seed: int = 42) -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    gen = SCMGenerator()
    taus, burned = [], []
    for spec in gen.grid(grid_config):
        scm = gen.generate(spec)
        tau = abs(_assoc_int_estimate(scm) - scm.true_int_ate())
        taus.append(tau)
        burned.append(spec.fingerprint())
    med = statistics.median(taus)
    delta_min = max(med, 0.05)  # frozen rule: max(calibration median, 0.05)
    report = {"n_instances": len(taus), "tau_median": med,
              "delta_min_frozen": delta_min,
              "tau_quantiles": {q: float(np.quantile(taus, q))
                                for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
              "floor_applied": delta_min > med + 1e-15 or med < 0.05}
    (out / "burned_list.json").write_text(json.dumps(sorted(burned), indent=2))
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2))
    return report


def run_frozen(frozen_config: dict, out_dir: str) -> dict:
    out = Path(out_dir)
    burned_path = out / "burned_list.json"
    burned = set(json.loads(burned_path.read_text())) if burned_path.exists() else set()
    gen = SCMGenerator()
    eg_values, m_preds, b_preds, truths = [], [], [], []
    for spec in gen.grid(frozen_config["grid"]):
        if spec.fingerprint() in burned:
            raise RuntimeError(
                f"BURNED-SET VIOLATION: spec {spec.fingerprint()} was used in "
                "calibration and is permanently excluded from the frozen phase")
        scm = gen.generate(spec)
        truth = scm.true_int_ate()
        method_pred = truth  # v0.1 oracle plumbing; real method in batch 02
        baseline_pred = _assoc_int_estimate(scm)
        m_err, b_err = abs(method_pred - truth), abs(baseline_pred - truth)
        near_zero = abs(truth) < frozen_config.get("near_zero_eps", 1e-3)
        eg = eg_score(m_err, b_err, near_zero=near_zero)
        if eg is not None:
            eg_values.append(eg)
        m_preds.append(method_pred); b_preds.append(baseline_pred); truths.append(truth)
    m_abs, b_abs = abs_errors(m_preds, truths), abs_errors(b_preds, truths)
    eg_arr = np.asarray(eg_values, float)
    report = {
        "delta_min": frozen_config["delta_min"],
        "eg_distribution": ({"mean": float(eg_arr.mean()),
                             "median": float(np.median(eg_arr)),
                             "q10": float(np.quantile(eg_arr, .1)),
                             "q90": float(np.quantile(eg_arr, .9)),
                             "n": int(eg_arr.size)} if eg_arr.size else None),
        "abs_errors": {"method": m_abs, "baseline": b_abs},
        "verdict": conjunctive_verdict(
            eg_better=bool(eg_arr.size and np.median(eg_arr) > 1.0),
            eg_sig=bool(eg_arr.size and np.quantile(eg_arr, .1) > 1.0),
            abs_better=m_abs["rmse"] < b_abs["rmse"],
            abs_sig=m_abs["rmse"] < 0.5 * b_abs["rmse"]),
        "scoping_note": "v0.1 method=oracle (pipeline check only); real MVP-2A "
                        "method + Q-C7 frozen grid land in task batch 02",
    }
    (out / "frozen_report.json").write_text(json.dumps(report, indent=2))
    return report
