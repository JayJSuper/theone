"""Minimal active-experiment loop (T2-05, 站桩册①, bet ③).

A learner estimates the hidden interventional ATE (slope beta_xy) of a 3-node
linear-Gaussian SCM by issuing do(X=x) queries under a fixed budget. Three point-
selection strategies are compared (registry X1):
  A = information-gain max (Bayesian experimental design; minimize posterior
      Var(beta) -> D-optimal, tends to the interval endpoints)
  B = uncertainty-weighted uniform coverage (HASH-B bucketing; remaining budget
      uniform, pre-registered)
  C = random passive

Pure simulation, no external side effects. All randomness seeded explicitly.
Criteria frozen in experiments/active_loop_min/prereg.md.

Honest scope: for slope estimation, D-optimal (A) beating random (C) is a classic
result - not a surprise. Bet ③'s hard half (B/C long-run SAFETY premium) is NOT
tested here.
"""
from __future__ import annotations
import numpy as np

X_LO, X_HI = -2.0, 2.0
ASSUMED_NOISE_VAR = 0.5 ** 2   # learner's working noise scale (does not see truth)
PRIOR_VAR = 10.0               # weak Gaussian prior on [intercept, slope]


class BayesLinReg:
    """Bayesian linear regression over w=[intercept, slope] with known noise var.
    Posterior precision A = (1/prior_var) I + (1/noise_var) sum phi phi^T."""

    def __init__(self):
        self.A = np.eye(2) / PRIOR_VAR          # precision
        self.b = np.zeros(2)                    # precision-weighted mean accumulator
        self.nv = ASSUMED_NOISE_VAR

    def update(self, x: float, y: float):
        phi = np.array([1.0, x])
        self.A += np.outer(phi, phi) / self.nv
        self.b += phi * y / self.nv

    def cov(self):
        return np.linalg.inv(self.A)

    def mean(self):
        return self.cov() @ self.b

    def slope_mean(self):
        return float(self.mean()[1])

    def var_slope_if_added(self, x: float) -> float:
        """Posterior Var(slope) if a query at x were added (design depends on x, not y)."""
        phi = np.array([1.0, x])
        A2 = self.A + np.outer(phi, phi) / self.nv
        return float(np.linalg.inv(A2)[1, 1])


def _truth(seed: int):
    rng = np.random.default_rng(seed)
    return {"beta_xy": float(rng.uniform(0.2, 0.8)),
            "beta_uy": float(rng.uniform(0.3, 0.7)),
            "sigma": float(rng.uniform(0.2, 0.5))}


def _query(truth, x, rng):
    """One do(X=x) observation: confounder U cut from X, so estimate is unbiased."""
    u = rng.standard_normal()
    return truth["beta_xy"] * x + truth["beta_uy"] * u + rng.standard_normal() * truth["sigma"]


def _select(strategy: str, model: BayesLinReg, counts, rng, n_buckets: int):
    grid = np.linspace(X_LO, X_HI, 41)
    if strategy == "A":                       # min posterior Var(slope) -> D-optimal
        vs = [model.var_slope_if_added(x) for x in grid]
        return float(grid[int(np.argmin(vs))])
    if strategy == "B":                       # least-covered bucket, center query
        edges = np.linspace(X_LO, X_HI, n_buckets + 1)
        j = int(np.argmin(counts))            # uniform coverage (remaining budget uniform)
        counts[j] += 1
        return float((edges[j] + edges[j + 1]) / 2)
    return float(rng.uniform(X_LO, X_HI))     # C: random passive


def run_one(strategy: str, seed: int, budget: int = 40, warmup: int = 2,
            n_buckets: int = 4):
    truth = _truth(seed)
    rng = np.random.default_rng(10_000 + seed)        # shared start across strategies
    model = BayesLinReg()
    counts = np.zeros(n_buckets)
    errs = []
    # shared warm-up points (fair start, identical across strategies)
    warm_rng = np.random.default_rng(seed)            # same for A/B/C given same seed
    for _ in range(warmup):
        x = float(warm_rng.uniform(X_LO, X_HI))
        model.update(x, _query(truth, x, rng))
        errs.append(abs(model.slope_mean() - truth["beta_xy"]))
    for _ in range(budget - warmup):
        x = _select(strategy, model, counts, rng, n_buckets)
        model.update(x, _query(truth, x, rng))
        errs.append(abs(model.slope_mean() - truth["beta_xy"]))
    return errs


def run_experiment(seeds=range(20), budget: int = 40):
    out = {"A": [], "B": [], "C": []}
    for s in seeds:
        for strat in ("A", "B", "C"):
            out[strat].append(run_one(strat, int(s), budget=budget))
    curves = {k: np.asarray(v) for k, v in out.items()}    # (n_seeds, budget)
    final = {k: curves[k][:, -1] for k in curves}          # error at budget
    # frozen vital test: paired sign test A < C across seeds
    a_better_c = int(np.sum(final["A"] < final["C"]))
    n = len(final["C"])
    results = {
        "budget": budget, "n_seeds": n,
        "final_error_median": {k: float(np.median(final[k])) for k in final},
        "final_error_iqr": {k: [float(np.quantile(final[k], .25)),
                                float(np.quantile(final[k], .75))] for k in final},
        "A_better_than_C_count": a_better_c,
        "vital_A_beats_C": bool(a_better_c >= 15),     # frozen threshold (>=15/20)
        "mean_curve": {k: [float(v) for v in curves[k].mean(0)] for k in curves},
    }
    return results
