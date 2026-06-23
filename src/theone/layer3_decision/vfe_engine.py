"""L3 · variational free energy (VFE) engine — minimize prediction error under a
generative model. Honest scope (fusion mandate answer 2/3): the test certifies that
the optimization CONVERGES (monotone descent to the analytic minimum), NOT that
'minimizing free energy = intelligence'. The credential claims only what is verified.

Generative model (linear-Gaussian, the verifiable core):
  observation o ~ N(W μ, I),  prior μ ~ N(0, I·1/β)
  F(μ) = ½‖o − W μ‖² + (β/2)‖μ‖²        (convex; unique minimum)
  μ* = (Wᵀ W + β I)⁻¹ Wᵀ o               (closed form — the independent recompute)
Gradient descent on F must (a) decrease monotonically and (b) reach μ* — checked
against the closed form, which is the recomputability gate.
"""
from __future__ import annotations
import numpy as np


class VFEEngine:
    def __init__(self, W: np.ndarray, beta: float = 1e-3, lr: float | None = None) -> None:
        self.W = np.asarray(W, dtype=float)
        self.beta = beta
        H = self.W.T @ self.W + beta * np.eye(self.W.shape[1])
        # stable step size from the curvature (largest eigenvalue of the Hessian)
        self.lr = lr if lr is not None else 1.0 / (np.max(np.linalg.eigvalsh(H)) + 1e-9)
        self._H = H

    def free_energy(self, mu: np.ndarray, o: np.ndarray) -> float:
        mu, o = np.asarray(mu, float), np.asarray(o, float)
        return float(0.5 * np.sum((o - self.W @ mu) ** 2) + 0.5 * self.beta * np.sum(mu ** 2))

    def grad(self, mu: np.ndarray, o: np.ndarray) -> np.ndarray:
        return self._H @ mu - self.W.T @ o

    def closed_form(self, o: np.ndarray) -> np.ndarray:
        return np.linalg.solve(self._H, self.W.T @ np.asarray(o, float))

    def minimize(self, o: np.ndarray, mu0: np.ndarray | None = None, iters: int = 500):
        """Gradient descent. Returns (mu_final, F_trace)."""
        o = np.asarray(o, float)
        mu = np.zeros(self.W.shape[1]) if mu0 is None else np.asarray(mu0, float).copy()
        trace = [self.free_energy(mu, o)]
        for _ in range(iters):
            mu = mu - self.lr * self.grad(mu, o)
            trace.append(self.free_energy(mu, o))
        return mu, np.array(trace)


__all__ = ["VFEEngine"]
