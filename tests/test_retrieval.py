"""Retrieval-kernel tests (M2 groundwork). Pins each kernel's real property;
no kernel is 'adopted' here — adoption needs a benchmark win vs cosine."""
import numpy as np
import pytest
from theone.memory.retrieval import (cosine_scores, spectral_scores, hrr_scores,
                                      hrr_bind, hrr_unbind, retrieve)

rng = np.random.default_rng(0)


def test_cosine_retrieves_exact_match():
    keys = rng.standard_normal((10, 16))
    q = keys[4].copy()
    assert retrieve(q, keys, "cosine", k=1)[0] == 4


def test_spectral_is_shift_invariant_cosine_is_not():
    """Path ① genuine property: magnitude spectrum ignores circular shift."""
    base = rng.standard_normal(32)
    shifted = np.roll(base, 11)
    distract = rng.standard_normal(32)
    keys = np.stack([shifted, distract])
    # spectral ranks the shifted copy first; cosine cannot tell it from noise
    assert retrieve(base, keys, "spectral", k=1)[0] == 0
    spec = spectral_scores(base, keys)
    assert spec[0] > 0.99                      # near-perfect spectral match
    assert abs(cosine_scores(base, keys)[0]) < 0.5   # cosine blind to the shift


def test_spectral_honest_weakness_low_margin_vs_random():
    """Honest: magnitude spectra of random vectors are broadly similar, so the
    margin over a random distractor is small (documented limitation)."""
    base = rng.standard_normal(64)
    distract = rng.standard_normal(64)
    s = spectral_scores(base, np.stack([np.roll(base, 5), distract]))
    assert s[0] - s[1] < 0.3                    # small margin, as audited


def test_hrr_bind_unbind_recovers_filler():
    """HRR core: bind(role, filler) then unbind(·, role) ~= filler (cosine high)."""
    role = _u(rng.standard_normal(256))
    filler = _u(rng.standard_normal(256))
    c = hrr_bind(role, filler)
    rec = hrr_unbind(c, role)
    cos = rec @ filler / (np.linalg.norm(rec) * np.linalg.norm(filler))
    assert cos > 0.4                            # noisy but clearly recovered


def test_retrieve_rejects_unknown_kernel():
    with pytest.raises(ValueError):
        retrieve(rng.standard_normal(8), rng.standard_normal((3, 8)), "magic")


def _u(v):
    return v / (np.linalg.norm(v) + 1e-9)
