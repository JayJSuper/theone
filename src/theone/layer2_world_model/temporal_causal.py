"""Temporal causal direction — verifiable lagged-causality test on continuous streams
(the perception->causal-direction seam for time series).

Attention is symmetric; causal direction is not (NOTE: the project's founding asymmetry).
For two streams we ask, in a Granger sense: does A's past improve the prediction of B's
present BEYOND B's own past — and is the reverse absent? A directed claim A->B is made
only when the forward improvement is significant AND the backward one is not (a clear
asymmetry). Symmetric or absent evidence -> ABSTAIN. The F-test statistic is the
recomputable credential.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import f as f_dist


def _lagged_gain(driver, target, lag: int = 1):
    """F-test: does `driver`'s lag improve the AR model of `target`? Returns (F, p, R2_gain)."""
    y = np.asarray(target, dtype=float)[lag:]
    tar_lag = np.asarray(target, dtype=float)[:-lag]
    drv_lag = np.asarray(driver, dtype=float)[:-lag]
    n = len(y)
    X_red = np.column_stack([np.ones(n), tar_lag])
    X_full = np.column_stack([np.ones(n), tar_lag, drv_lag])

    def _rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r)
    rss_r, rss_f = _rss(X_red), _rss(X_full)
    df1, df2 = 1, n - X_full.shape[1]
    F = ((rss_r - rss_f) / df1) / (rss_f / df2 + 1e-18)
    p = float(f_dist.sf(F, df1, df2))
    r2_gain = (rss_r - rss_f) / (rss_r + 1e-18)
    return float(F), p, float(r2_gain)


def temporal_direction(a, b, lag: int = 1, alpha: float = 1e-3) -> dict:
    """Directed verdict between streams a and b. ANSWER a->b or b->a only on a clear
    asymmetry; otherwise ABSTAIN (ambiguous / symmetric / absent)."""
    F_ab, p_ab, g_ab = _lagged_gain(a, b, lag)   # a -> b
    F_ba, p_ba, g_ba = _lagged_gain(b, a, lag)   # b -> a
    fwd = p_ab < alpha
    bwd = p_ba < alpha
    if fwd and not bwd:
        verdict = "a->b"
    elif bwd and not fwd:
        verdict = "b->a"
    else:
        verdict = "abstain"
    return {"verdict": verdict, "p_ab": p_ab, "p_ba": p_ba,
            "F_ab": round(F_ab, 3), "F_ba": round(F_ba, 3),
            "r2_gain_ab": round(g_ab, 4), "r2_gain_ba": round(g_ba, 4)}


__all__ = ["temporal_direction"]
