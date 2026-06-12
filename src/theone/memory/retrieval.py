"""Memory-retrieval kernels (M2 groundwork) — bake-off candidates from the
seven-paths audit, reimplemented + tested (their '预期输出' claims demoted to
design until they pass a benchmark vs the cosine baseline).

  cosine    — plain semantic similarity (the baseline to beat)
  spectral  — Fourier-magnitude match (path ①); SHIFT-INVARIANT (verified)
  hrr       — Holographic Reduced Representation binding/similarity (path ④)

Pure numpy, no new deps. A retrieval kernel scores a query against keys and
returns ranked indices. Adoption rule: a kernel enters The One only if it beats
cosine on a frozen benchmark — not because it sounds profound.
"""
from __future__ import annotations
import numpy as np
from numpy.fft import fft, ifft


def _unit(v, axis=-1, eps=1e-9):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def cosine_scores(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Plain cosine similarity. query (d,), keys (N,d) -> (N,)."""
    return _unit(keys, axis=1) @ _unit(query)


def spectral_scores(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Fourier-magnitude cosine (path ①). Invariant to circular shift of the
    pattern (magnitude spectrum ignores phase) — its genuine differentiator."""
    qf = np.abs(fft(query))
    kf = np.abs(fft(keys, axis=1))
    return _unit(kf, axis=1) @ _unit(qf)


# ----- Holographic Reduced Representation (Plate 1995) ---------------------
def hrr_bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Circular convolution: bind two vectors into one of the same dim."""
    return np.real(ifft(fft(a) * fft(b)))


def hrr_unbind(c: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Approximate inverse of bind: correlate c with a to recover b."""
    return np.real(ifft(fft(c) * np.conj(fft(a))))


def hrr_scores(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """HRR similarity = cosine in the circular-convolution algebra. For raw
    vectors this reduces toward cosine; its power is on BOUND structures
    (role-filler pairs), exercised in the bake-off, not here."""
    return _unit(keys, axis=1) @ _unit(query)


KERNELS = {"cosine": cosine_scores, "spectral": spectral_scores, "hrr": hrr_scores}


def retrieve(query: np.ndarray, keys: np.ndarray, kernel: str = "cosine",
             k: int = 1) -> np.ndarray:
    """Return indices of the top-k keys under the named kernel (desc score)."""
    if kernel not in KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}; have {list(KERNELS)}")
    scores = KERNELS[kernel](np.asarray(query, float), np.asarray(keys, float))
    return np.argsort(-scores)[:k]
