"""Identification estimators that COMPOSE the frozen T1 engine's public queries
(engine.py untouched). Front-door estimation (Q-C14): recover P(Y=1|do(X)) using
ONLY observed {X, M, Y} quantities, even when the X<->Y confounder is unobserved.

Front-door formula (Pearl):
    P(Y|do(X=x)) = sum_m P(M=m|X=x) * sum_x' P(Y|X=x', M=m) P(X=x')
"""
from __future__ import annotations
from .engine import InterventionEngine


def frontdoor_prob(engine: InterventionEngine, X: str, Y: str, M: str,
                   y_val=1, x_do=1) -> float:
    """P(Y=y_val | do(X=x_do)) via the front-door adjustment, observed-only."""
    g = engine.g
    m_states = list(g.states(M))
    x_states = list(g.states(X))
    total = 0.0
    for m in m_states:
        p_m_given_x = engine.query_observation(M, m, {X: x_do}).value
        inner = 0.0
        for xp in x_states:
            p_y = engine.query_observation(Y, y_val, {X: xp, M: m}).value
            p_xp = engine.query_observation(X, xp, {}).value
            inner += p_y * p_xp
        total += p_m_given_x * inner
    return total


def frontdoor_ate(engine: InterventionEngine, X: str, Y: str, M: str) -> float:
    """Front-door estimate of the interventional ATE on Y==1 (do X=1 vs do X=0)."""
    return (frontdoor_prob(engine, X, Y, M, 1, 1)
            - frontdoor_prob(engine, X, Y, M, 1, 0))
