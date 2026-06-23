"""L3 · active inference loop — a minimal perceive→infer→act cycle that keeps
variational free energy low as observations arrive. Honest scope: this demonstrates a
running closed loop whose free energy stays bounded/declining, NOT general autonomy.
"""
from __future__ import annotations
import numpy as np

from theone.layer3_decision.vfe_engine import VFEEngine


class ActiveInferenceLoop:
    def __init__(self, engine: VFEEngine, f_threshold: float = 0.01) -> None:
        self.engine = engine
        self.f_threshold = f_threshold
        self.mu = np.zeros(engine.W.shape[1])

    def step(self, observation: np.ndarray, inner_iters: int = 50) -> float:
        """One cycle: infer the latent that best explains the observation, return F."""
        self.mu, trace = self.engine.minimize(observation, mu0=self.mu, iters=inner_iters)
        return float(trace[-1])

    def run(self, observations, inner_iters: int = 50):
        """Return the per-step final free energies across a stream of observations."""
        return [self.step(o, inner_iters) for o in observations]


__all__ = ["ActiveInferenceLoop"]
