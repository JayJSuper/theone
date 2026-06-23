"""L1 · SSM continuous encoder — a stable linear state-space model that encodes a
continuous signal x(t) into a latent trajectory h(t), reconstructable by a LINEAR
decoder. No tokenizer: the world enters as a continuous stream, O(N) in length.

Design (echo-state / reservoir form, fully verifiable):
  h[t] = A h[t-1] + B x[t]        (A rescaled to spectral radius < 1 → stable)
  x_hat[t] = C h[t]              (D = 0: reconstruction MUST flow through the state,
                                  so h genuinely encodes the signal — not a passthrough)
C is fit by least squares (the linear readout). A bank of decaying modes at mixed
timescales lets the readout reconstruct a smooth signal to MSE < 1e-3. Spectral
radius < 1 is the stability invariant Layer 0/credential checks.
"""
from __future__ import annotations
import numpy as np


class SSMEncoder:
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64,
                 spectral_radius: float = 0.9, seed: int = 0) -> None:
        if not (0.0 < spectral_radius < 1.0):
            raise ValueError("spectral_radius must be in (0, 1) for stability")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((hidden_dim, hidden_dim))
        radius = max(np.abs(np.linalg.eigvals(A)))
        self.A = A * (spectral_radius / radius)          # rescale to target radius
        self.B = rng.standard_normal((hidden_dim, input_dim)) * 0.5
        self.C: np.ndarray | None = None                 # fit by fit_decoder

    @property
    def spectral_radius(self) -> float:
        return float(max(np.abs(np.linalg.eigvals(self.A))))

    def encode(self, x: np.ndarray) -> np.ndarray:
        """x: (T, input_dim) → H: (T, hidden_dim). The continuous latent trajectory."""
        x = np.atleast_2d(x)
        if x.shape[0] == 1 and x.shape[1] != self.input_dim:
            x = x.reshape(-1, self.input_dim)
        T = x.shape[0]
        H = np.zeros((T, self.hidden_dim))
        h = np.zeros(self.hidden_dim)
        for t in range(T):
            h = self.A @ h + self.B @ x[t]
            H[t] = h
        return H

    def fit_decoder(self, x: np.ndarray, H: np.ndarray, ridge: float = 1e-8) -> None:
        """Least-squares (ridge) readout C: argmin ||x - H Cᵀ||."""
        x = x.reshape(-1, self.input_dim)
        G = H.T @ H + ridge * np.eye(self.hidden_dim)
        self.C = np.linalg.solve(G, H.T @ x).T          # (input_dim, hidden_dim)

    def decode(self, H: np.ndarray) -> np.ndarray:
        if self.C is None:
            raise RuntimeError("decoder not fit; call fit_decoder first")
        return H @ self.C.T

    def reconstruction_mse(self, x: np.ndarray, H: np.ndarray | None = None) -> float:
        x = x.reshape(-1, self.input_dim)
        H = self.encode(x) if H is None else H
        xhat = self.decode(H)
        return float(np.mean((x - xhat) ** 2))


__all__ = ["SSMEncoder"]
