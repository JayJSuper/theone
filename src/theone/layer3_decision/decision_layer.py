"""L3 DecisionLayer — the VFE engine as a CredentialedLayer.

Admissibility: free energy descended monotonically AND reached the threshold.
Recomputability: the gradient-descent minimum is checked against the closed-form
solution μ* (two independent routes to the same optimum). The regime declares the
honest limit — convergence on a given convex objective, not 'autonomy'.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer3_decision.vfe_engine import VFEEngine


class DecisionLayer(CredentialedLayer):
    name = "L3_decision"
    layer_index = 3
    F_THRESHOLD = 0.01

    def process(self, inputs: Any) -> LayerVerdict:
        W = np.asarray(inputs["W"], dtype=float)
        o = np.asarray(inputs["observation"], dtype=float)
        engine = VFEEngine(W, beta=inputs.get("beta", 1e-3))
        mu, trace = engine.minimize(o, iters=inputs.get("iters", 500))

        monotone = bool(np.all(np.diff(trace) <= 1e-9))
        if not monotone:
            return LayerVerdict.abstain(self.name, "free energy not monotone-decreasing")
        if trace[-1] > self.F_THRESHOLD:
            return LayerVerdict.abstain(
                self.name, f"free energy {trace[-1]:.3f} did not reach {self.F_THRESHOLD}")

        # recomputability gate: gradient-descent μ vs closed-form μ*
        mu_star = engine.closed_form(o)

        def _recompute():
            return round(engine.free_energy(mu_star, o), 9)

        cred = Credential(
            self.name, claim="active inference minimized free energy to its optimum",
            value=round(float(trace[-1]), 9),
            regime="convergence on a convex generative objective (not general autonomy)",
            recompute=_recompute, tolerance=1e-6,
            evidence={"f_initial": float(trace[0]), "iters": len(trace) - 1,
                      "mu_vs_closedform": float(np.max(np.abs(mu - mu_star))),
                      "f_threshold": self.F_THRESHOLD})
        return LayerVerdict.answer(self.name, cred, value={**inputs, "mu": mu, "f_final": float(trace[-1])})


__all__ = ["DecisionLayer"]
