"""L0 · PINN physics-residual constraint (numpy collocation form, no autodiff needed).

A flexible model (polynomial basis) is fit to noisy trajectory observations. WITHOUT a
physics prior it overfits and extrapolation explodes; adding a physics-residual penalty
that enforces the ODE  q'' + omega^2 q = 0  at collocation points over a wider window
pins the solution to the physical law, so it extrapolates. Honest scope (fusion answer 2):
valid only where the system genuinely obeys the stated ODE; the benefit is measured, not
assumed. Demonstrated on the simple harmonic oscillator.
"""
from __future__ import annotations
import numpy as np


class PINNConstraint:
    def __init__(self, omega: float = 1.0, degree: int = 12) -> None:
        self.omega = omega
        self.degree = degree
        self.coef_: np.ndarray | None = None

    def _design(self, t: np.ndarray) -> np.ndarray:
        return np.vander(np.asarray(t, dtype=float), self.degree + 1, increasing=True)

    def _second_deriv_design(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        k = np.arange(self.degree + 1)
        V2 = np.zeros((len(t), self.degree + 1))
        for j, tt in enumerate(t):
            V2[j] = [(kk * (kk - 1) * tt ** (kk - 2) if kk >= 2 else 0.0) for kk in k]
        return V2

    def fit(self, t_train, q_train, collocation=None, lam: float = 5.0) -> "PINNConstraint":
        A = self._design(t_train)
        if lam > 0:
            tc = (np.linspace(float(np.min(t_train)), float(np.max(t_train)) * 1.75, 120)
                  if collocation is None else np.asarray(collocation, dtype=float))
            phys = self._second_deriv_design(tc) + self.omega ** 2 * self._design(tc)
            A = np.vstack([A, lam * phys])
            b = np.concatenate([np.asarray(q_train, dtype=float), np.zeros(len(tc))])
        else:
            b = np.asarray(q_train, dtype=float)
        self.coef_ = np.linalg.lstsq(A, b, rcond=None)[0]
        return self

    def predict(self, t) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("call fit first")
        return self._design(t) @ self.coef_


def extrapolation_benefit(omega=1.0, degree=12, seed=0) -> dict:
    """Measured extrapolation RMSE: data-only vs physics-constrained, on the SHO."""
    rng = np.random.default_rng(seed)
    t_tr = np.linspace(0, 4, 60)
    q_tr = np.cos(omega * t_tr) + rng.normal(0, 0.02, 60)
    t_ex = np.linspace(4, 7, 60)
    q_ex = np.cos(omega * t_ex)
    data_only = PINNConstraint(omega, degree).fit(t_tr, q_tr, lam=0.0)
    physics = PINNConstraint(omega, degree).fit(t_tr, q_tr, lam=5.0)
    e_data = float(np.sqrt(np.mean((data_only.predict(t_ex) - q_ex) ** 2)))
    e_pinn = float(np.sqrt(np.mean((physics.predict(t_ex) - q_ex) ** 2)))
    return {"rmse_data_only": e_data, "rmse_physics": e_pinn,
            "improvement": 1.0 - e_pinn / e_data if e_data > 0 else 0.0}


__all__ = ["PINNConstraint", "extrapolation_benefit"]
