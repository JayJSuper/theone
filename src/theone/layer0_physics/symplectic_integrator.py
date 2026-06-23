"""L0 · symplectic integrator — time-evolution that preserves phase-space structure,
so total energy stays bounded over long horizons (a verifiable invariant). Contrast:
explicit Euler injects energy and diverges.

Worldview (held boldly, per the fusion mandate): physical-dynamical structure may be
the right substrate for a cognitive latent. Verification (held honestly): we only ever
*claim* what the energy-drift test certifies — on systems whose latent genuinely has
Hamiltonian structure. The integrator is the instrument; the regime field is the limit.

Hamiltonian (simple harmonic oscillator): H(q,p) = ½ p² + ½ ω² q².
"""
from __future__ import annotations
import numpy as np


def sho_energy(q, p, omega: float = 1.0) -> float:
    return float(0.5 * np.sum(np.asarray(p) ** 2) + 0.5 * omega ** 2 * np.sum(np.asarray(q) ** 2))


class SymplecticIntegrator:
    """Velocity-Verlet (leapfrog) — a 2nd-order symplectic method. For the SHO the
    force is F(q) = -ω² q."""

    def __init__(self, omega: float = 1.0, dt: float = 0.01) -> None:
        self.omega = omega
        self.dt = dt

    def _force(self, q):
        return -(self.omega ** 2) * np.asarray(q)

    def step(self, q, p):
        q = np.asarray(q, dtype=float); p = np.asarray(p, dtype=float)
        a = self._force(q)
        p_half = p + 0.5 * self.dt * a
        q_new = q + self.dt * p_half
        a_new = self._force(q_new)
        p_new = p_half + 0.5 * self.dt * a_new
        return q_new, p_new

    def evolve(self, q0, p0, steps: int):
        """Return (trajectory of energies, final (q, p))."""
        q = np.atleast_1d(np.asarray(q0, dtype=float))
        p = np.atleast_1d(np.asarray(p0, dtype=float))
        energies = [sho_energy(q, p, self.omega)]
        for _ in range(steps):
            q, p = self.step(q, p)
            energies.append(sho_energy(q, p, self.omega))
        return np.array(energies), (q, p)


class ExplicitEuler:
    """Reference non-symplectic integrator — energy grows without bound (the foil)."""

    def __init__(self, omega: float = 1.0, dt: float = 0.01) -> None:
        self.omega = omega
        self.dt = dt

    def evolve(self, q0, p0, steps: int):
        q = np.atleast_1d(np.asarray(q0, dtype=float))
        p = np.atleast_1d(np.asarray(p0, dtype=float))
        energies = [sho_energy(q, p, self.omega)]
        for _ in range(steps):
            a = -(self.omega ** 2) * q
            q_new = q + self.dt * p
            p_new = p + self.dt * a
            q, p = q_new, p_new
            energies.append(sho_energy(q, p, self.omega))
        return np.array(energies), (q, p)


__all__ = ["SymplecticIntegrator", "ExplicitEuler", "sho_energy"]
