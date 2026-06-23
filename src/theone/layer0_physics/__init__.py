"""Layer 0 · physical-constraint dynamics — the verifiable invariant floor.

Worldview held boldly: physical-dynamical structure may be the right substrate for a
cognitive latent. Verification held honestly: the symplectic integrator + energy
monitor form a constraint-credential generator; the PhysicsLayer ANSWERs only when
energy drift is certified under threshold, and the credential's regime declares the
limit. The math is real and tested; the metaphysics stays an open hypothesis.
"""
from theone.layer0_physics.symplectic_integrator import (
    SymplecticIntegrator, ExplicitEuler, sho_energy,
)
from theone.layer0_physics.energy_monitor import EnergyMonitor
from theone.layer0_physics.physics_layer import PhysicsLayer
from theone.layer0_physics.pinn_constraint import PINNConstraint, extrapolation_benefit

__all__ = ["SymplecticIntegrator", "ExplicitEuler", "sho_energy",
           "EnergyMonitor", "PhysicsLayer", "PINNConstraint", "extrapolation_benefit"]
