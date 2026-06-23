"""The One · native perception front-end — continuous streams -> a confounder the native
engine can verifiably adjust on.

Connects B3 (O(N) SSM encoder) to the native verifiable engine: a latent confounder is
observed only as noisy continuous SEQUENCES; the SSM integrates each sequence (O(N),
denoising over time) into a summary, which is discretized into strata the engine adjusts
on. So verifiable causal inference can run on CONTINUOUS PERCEPTION, not just tabular
confounders — and longer sequences -> better perception -> more accurate do().

Honest scope: linear SSM summary + median/quantile discretization; a feasibility bridge,
not a tuned perceptual model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.layer1_perception.ssm_encoder import SSMEncoder


class SSMPerception:
    def __init__(self, hidden_dim: int = 24, seed: int = 0) -> None:
        self.enc = SSMEncoder(input_dim=1, hidden_dim=hidden_dim, seed=seed)

    def summarize(self, streams: np.ndarray) -> np.ndarray:
        """streams: (n_units, T). Return a 1-D per-unit confounder summary via the SSM
        (mean hidden state projected onto its dominant direction = denoised confounder)."""
        feats = np.array([self.enc.encode(s.reshape(-1, 1)).mean(axis=0) for s in streams])
        fc = feats - feats.mean(0)
        # project onto the top principal direction (the shared confounder signal)
        u, s, vt = np.linalg.svd(fc, full_matrices=False)
        proj = fc @ vt[0]
        return proj

    def to_confounder(self, streams: np.ndarray, n_strata: int = 2) -> np.ndarray:
        """Discretize the SSM summary into n_strata equal-frequency strata (labels 0..n-1)."""
        proj = self.summarize(streams)
        qs = np.quantile(proj, np.linspace(0, 1, n_strata + 1)[1:-1])
        return np.digitize(proj, qs).astype(int)

    def perceive_into_df(self, streams: np.ndarray, x: np.ndarray, y: np.ndarray,
                         n_strata: int = 2) -> pd.DataFrame:
        """Build the DataFrame the native engine consumes, with a PERCEIVED confounder U."""
        u_hat = self.to_confounder(streams, n_strata)
        return pd.DataFrame({"U": u_hat, "X": x.astype(int), "Y": y.astype(int)})

    def perceive_features(self, streams: np.ndarray, k: int = 4) -> np.ndarray:
        """Return the top-k SSM principal components as CONTINUOUS covariates (n_units, k) —
        the perception front-end for the continuous native path (estimate_continuous), where
        the confounder is adjusted on as continuous features rather than discretized strata."""
        feats = np.array([self.enc.encode(s.reshape(-1, 1)).mean(axis=0) for s in streams])
        fc = feats - feats.mean(0)
        u, s, vt = np.linalg.svd(fc, full_matrices=False)
        kk = min(k, vt.shape[0])
        return (fc @ vt[:kk].T).astype(np.float32)        # (n_units, k) continuous features


__all__ = ["SSMPerception"]
