"""L1 PerceptionLayer — the SSM encoder as a CredentialedLayer.

Encodes a continuous signal into a latent trajectory and ANSWERs only if BOTH gates
pass: admissibility (stable: spectral radius < 1, AND faithful: reconstruction MSE <
tol) and recomputability (the credential re-encodes + re-decodes from the stored
A/B/C and reproduces the MSE). The latent H is passed forward as the L1→L2 connector
(the seam the native_causal_latent probes prototyped).
"""
from __future__ import annotations
from typing import Any
import numpy as np

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer1_perception.ssm_encoder import SSMEncoder


class PerceptionLayer(CredentialedLayer):
    name = "L1_perception"
    layer_index = 1
    MSE_TOL = 1e-3

    def __init__(self, hidden_dim: int = 64, spectral_radius: float = 0.9, seed: int = 0):
        self.hidden_dim = hidden_dim
        self.spectral_radius = spectral_radius
        self.seed = seed

    def process(self, inputs: Any) -> LayerVerdict:
        x = np.asarray(inputs["signal"], dtype=float).reshape(-1, inputs.get("input_dim", 1))
        if not np.all(np.isfinite(x)):          # degenerate input → refuse, don't emit NaN latent
            return LayerVerdict.abstain(self.name, "signal contains non-finite values (NaN/inf)")
        try:
            enc = SSMEncoder(input_dim=x.shape[1], hidden_dim=self.hidden_dim,
                             spectral_radius=self.spectral_radius, seed=self.seed)
        except ValueError as e:
            return LayerVerdict.abstain(self.name, f"unstable encoder requested: {e}")

        H = enc.encode(x)
        enc.fit_decoder(x, H)
        mse = enc.reconstruction_mse(x, H)
        rho = enc.spectral_radius
        if rho >= 1.0:
            return LayerVerdict.abstain(self.name, f"unstable: spectral radius {rho:.3f} >= 1")
        if mse > self.MSE_TOL:
            return LayerVerdict.abstain(self.name, f"reconstruction MSE {mse:.2e} > {self.MSE_TOL}")

        cred = Credential(
            self.name, claim="SSM encoding is stable and reconstructs the signal",
            value=round(mse, 15), regime="linear-decoder reconstruction; spectral radius < 1",
            recompute=lambda: round(enc.reconstruction_mse(x), 15),
            tolerance=1e-9,
            evidence={"spectral_radius": rho, "hidden_dim": self.hidden_dim,
                      "n_steps": int(x.shape[0]), "mse_threshold": self.MSE_TOL})
        # pass the latent trajectory forward — the L1->L2 connector
        return LayerVerdict.answer(self.name, cred,
                                   value={**inputs, "latent": H, "encoder": enc})


__all__ = ["PerceptionLayer"]
