"""SCM generator (CC-07) + EG metrics & A7 statistical judgment (CC-08).

Frozen dependencies (registry):
- Q-C5: confounding strength = standardized coefficient product beta_ux * beta_uy.
- Amendment 1/1a: EG and absolute errors (RMSE/MAE) are JOINT primary metrics;
  conjunctive verdict; near-zero-effect instances use absolute error only.
- A7 final form (Q-C6): are_different=True iff (1) BCa 95% CI of tau excludes 0
  AND (2) |tau| >= delta_min (calibrated, frozen). Permutation = auxiliary only
  (strong null caveat). delta_min=None => CalibrationRequiredError.
- Grid VALUES are injected via config, never hard-coded (pending Q-C7 freeze).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from statistics import NormalDist
import hashlib
import json
import numpy as np
from ..types import CompareResult, TheOneConfig, CalibrationRequiredError

_ND = NormalDist()


# ====================== CC-07: SCM generator ==========================
@dataclass(frozen=True)
class SCMSpec:
    graph_type: str          # "linear_gaussian_3node"
    beta_ux: float
    beta_uy: float
    beta_xy: float
    noise: float             # extra noise scale on Y (>=0)
    n_samples: int
    seed: int

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class SyntheticSCM:
    """Standardized linear-Gaussian three-node SCM: U->X, U->Y, X->Y.
    Analytic truths: int ATE (unit do-shift) = beta_xy;
    obs slope = beta_xy + beta_ux*beta_uy (confounding bias = coefficient product, Q-C5)."""

    def __init__(self, spec: SCMSpec) -> None:
        if spec.graph_type != "linear_gaussian_3node":
            raise ValueError(f"unknown graph_type {spec.graph_type}")
        self.spec = spec

    def true_int_ate(self) -> float:
        return self.spec.beta_xy

    def true_obs_slope(self) -> float:
        return self.spec.beta_xy + self.spec.beta_ux * self.spec.beta_uy

    def true_bias(self) -> float:
        return self.spec.beta_ux * self.spec.beta_uy

    def sample(self, n: int | None = None, seed: int | None = None) -> dict:
        s = self.spec
        n = n or s.n_samples
        rng = np.random.default_rng(s.seed if seed is None else seed)
        u = rng.standard_normal(n)
        ex_var = max(1e-9, 1 - s.beta_ux ** 2)
        x = s.beta_ux * u + rng.standard_normal(n) * math.sqrt(ex_var)
        ey = rng.standard_normal(n) * max(s.noise, 1e-9)
        y = s.beta_xy * x + s.beta_uy * u + ey
        return {"U": u, "X": x, "Y": y}


class SCMGenerator:
    def generate(self, spec: SCMSpec) -> SyntheticSCM:
        return SyntheticSCM(spec)

    def grid(self, grid_config: dict):
        """Grid values come from grid_config (frozen pre-registration, Q-C7).
        No values are hard-coded here by design."""
        base_seed = int(grid_config.get("base_seed", 42))
        n = int(grid_config["n_samples"])
        i = 0
        for bux in grid_config["beta_ux"]:
            for buy in grid_config["beta_uy"]:
                for bxy in grid_config["beta_xy"]:
                    for noise in grid_config["noise"]:
                        yield SCMSpec("linear_gaussian_3node", float(bux), float(buy),
                                      float(bxy), float(noise), n, base_seed + i)
                        i += 1


# ====================== CC-08: metrics ================================
def abs_errors(preds, truths) -> dict:
    p, t = np.asarray(preds, float), np.asarray(truths, float)
    return {"rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "mae": float(np.mean(np.abs(p - t)))}


def eg_score(method_err: float, baseline_err: float, near_zero: bool = False,
             eps: float = 1e-12) -> float | None:
    """EG = baseline intervention error / method intervention error (>1 => method
    better). Amendment 1: for near-zero true effects EG is NOT normalized -
    return None and judge on absolute error instead."""
    if near_zero:
        return None
    return float(baseline_err) / max(float(method_err), eps)


def conjunctive_verdict(eg_better: bool | None, eg_sig: bool,
                        abs_better: bool, abs_sig: bool) -> str:
    """Amendment 1a: 'effective' requires SAME direction AND both significant;
    conflict => 'inconsistent evidence', no directional claim allowed."""
    if eg_better is None:  # near-zero regime: absolute error decides alone
        return "effective" if (abs_better and abs_sig) else "not_supported"
    if eg_better and abs_better and eg_sig and abs_sig:
        return "effective"
    if (not eg_better) and (not abs_better) and eg_sig and abs_sig:
        return "negative"
    return "inconsistent_evidence"


# ---------------------- bootstrap machinery ---------------------------
def _bca_interval(theta_hat: float, boot: np.ndarray, jack: np.ndarray,
                  alpha: float) -> tuple:
    """Bias-corrected & accelerated bootstrap CI."""
    boot = np.asarray(boot, float)
    prop = np.clip(np.mean(boot < theta_hat), 1e-9, 1 - 1e-9)
    z0 = _ND.inv_cdf(prop)
    jm = jack.mean()
    num = np.sum((jm - jack) ** 3)
    den = 6.0 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = 0.0 if den == 0 else num / den
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = _ND.inv_cdf(q)
        adj = _ND.cdf(z0 + (z0 + z) / max(1e-12, (1 - a * (z0 + z))))
        out.append(float(np.quantile(boot, np.clip(adj, 0.0, 1.0))))
    return out[0], out[1]


def _ate_obs(xy: np.ndarray) -> float:
    """Observational ATE for binary x,y rows [[x,y],...]."""
    x, y = xy[:, 0], xy[:, 1]
    m1, m0 = y[x == 1], y[x == 0]
    if len(m1) == 0 or len(m0) == 0:
        return float("nan")
    return float(m1.mean() - m0.mean())


def a7_judgment(obs_xy, int_y_by_arm: dict, config: TheOneConfig) -> CompareResult:
    """Frozen A7 conjunction on finite samples.
    obs_xy: array-like of (x,y) pairs from the observational regime.
    int_y_by_arm: {1: y-array under do(X=1), 0: y-array under do(X=0)}.
    """
    if config.delta_min is None:
        raise CalibrationRequiredError(
            "delta_min is uncalibrated; run the calibration phase first "
            "(frozen rule: A7 must not run with an arbitrary substantive threshold)")
    rng = np.random.default_rng(config.seed)
    obs = np.asarray(obs_xy, float)
    y1, y0 = (np.asarray(int_y_by_arm[1], float), np.asarray(int_y_by_arm[0], float))

    def tau(o, a1, a0):
        return _ate_obs(o) - (a1.mean() - a0.mean())

    tau_hat = tau(obs, y1, y0)
    # bootstrap
    B = config.bootstrap_B
    boot = np.empty(B)
    for b in range(B):
        oi = rng.integers(0, len(obs), len(obs))
        i1 = rng.integers(0, len(y1), len(y1))
        i0 = rng.integers(0, len(y0), len(y0))
        boot[b] = tau(obs[oi], y1[i1], y0[i0])
    # jackknife (over observational rows; arms are conditionally tight)
    jack = np.array([tau(np.delete(obs, i, axis=0), y1, y0)
                     for i in range(min(len(obs), 200))])
    lo, hi = _bca_interval(tau_hat, boot, jack, config.alpha)

    cond1 = not (lo <= 0.0 <= hi)
    cond2 = abs(tau_hat) >= config.delta_min
    if cond1 and cond2:
        verdict: bool | str = True
    elif not cond1:
        verdict = False
    else:
        verdict = "statistically_significant_below_substantive_threshold"

    # auxiliary permutation (STRONG null: full distribution equality across
    # regimes; under confounding the regimes differ beyond ATE - report only)
    pooled = np.concatenate([obs, np.column_stack([np.ones_like(y1), y1]),
                             np.column_stack([np.zeros_like(y0), y0])])
    n_obs = len(obs)
    perm_stats = np.empty(200)
    for b in range(200):
        idx = rng.permutation(len(pooled))
        o = pooled[idx[:n_obs]]
        rest = pooled[idx[n_obs:]]
        a1 = rest[rest[:, 0] == 1][:, 1]
        a0 = rest[rest[:, 0] == 0][:, 1]
        perm_stats[b] = (tau(o, a1, a0)
                         if len(a1) and len(a0) else np.nan)
    perm_stats = perm_stats[~np.isnan(perm_stats)]
    p_perm = float(np.mean(np.abs(perm_stats) >= abs(tau_hat))) if len(perm_stats) else None

    obs_ate = _ate_obs(obs)
    int_ate = float(y1.mean() - y0.mean())
    return CompareResult(
        obs_ate=obs_ate, int_ate=int_ate, are_different=verdict,
        stats={"tau_hat": float(tau_hat), "bca_ci": (lo, hi),
               "delta_min": config.delta_min, "cond_ci_excludes_0": cond1,
               "cond_substantive": cond2,
               "permutation_p_auxiliary_strong_null": p_perm,
               "mode": "a7_frozen_conjunction"})


def a7_from_summary(ci_low: float, ci_high: float, tau_abs: float,
                    delta_min: float):
    """Frozen C6 self-check fixed cases: judgment from precomputed summaries."""
    cond1 = not (ci_low <= 0.0 <= ci_high)
    cond2 = tau_abs >= delta_min
    if cond1 and cond2:
        return True
    if not cond1:
        return False
    return "statistically_significant_below_substantive_threshold"
