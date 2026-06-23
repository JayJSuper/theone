"""Latent-confounding sensitivity — turning 'latent confounding UNCERTIFIED' from a
warning into a recomputable QUANTITATIVE bound (VanderWeele & Ding E-value).

The engine computes do() exactly GIVEN the modeled structure; it cannot rule out an
UNMODELED confounder (NOTE-004 / discovery leg). The E-value quantifies exactly how
strong such a confounder would have to be: it is the minimum association (risk-ratio
scale) that an unmeasured confounder would need with BOTH treatment and outcome to fully
explain away the observed interventional contrast. A large E-value ⇒ the conclusion is
robust (only an implausibly strong hidden confounder could overturn it); an E-value near
1 ⇒ fragile. This is the recomputable handle on the part discovery cannot certify.
"""
from __future__ import annotations
import numpy as np


def e_value_rr(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele-Ding). Symmetric for protective effects."""
    if rr <= 0:
        return float("inf")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def e_value_for_do(p_do1: float, p_do0: float) -> dict:
    """E-value for an interventional contrast P(Y=1|do(X=1)) vs P(Y=1|do(X=0))."""
    p1 = min(max(p_do1, 1e-12), 1 - 1e-12)
    p0 = min(max(p_do0, 1e-12), 1 - 1e-12)
    rr = p1 / p0
    return {"risk_ratio": round(rr, 6), "e_value": round(e_value_rr(rr), 4),
            "interpretation": ("an unmeasured confounder would need associations >= this "
                               "E-value (risk-ratio scale) with BOTH treatment and outcome "
                               "to fully explain away the effect")}


def e_value_continuous(ate: float, outcome_sd: float) -> dict:
    """E-value for a CONTINUOUS outcome (VanderWeele & Ding): convert the standardized mean
    difference d = ATE / sd(Y) to an approximate risk ratio RR ≈ exp(0.91·d), then E-value.
    Lets the three-zone classifier handle continuous outcomes (IHDP-grade real data)."""
    sd = max(abs(outcome_sd), 1e-9)
    d = ate / sd
    rr = float(np.exp(0.91 * d))
    return {"std_effect": round(d, 4), "approx_risk_ratio": round(rr, 4),
            "e_value": round(e_value_rr(rr), 4)}


__all__ = ["e_value_rr", "e_value_for_do", "e_value_continuous"]
