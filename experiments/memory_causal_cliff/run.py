"""Pillar 2 (sovereign memory) — the memory analogue of the cliff, and the
cross-pillar unification. WEAK-02 found causal-aware retrieval ties flat cosine on
AVERAGE accuracy; its value is safety, not average. Here we show the regime where
that safety bites: retrieving a past memory to inform a CAUSAL decision.

Each memory stores a past intervention with a true (back-door-adjusted) causal
effect c_m and a surface ASSOCIATION a_m = c_m + confounding shift. A new query
needs the causal effect c_q of an analogous intervention; we retrieve a memory and
transfer its c_m.
  - flat (cosine/embedding) retrieval: match on ASSOCIATION (surface similarity) ->
    returns the associationally-nearest memory, whose causal effect can be far off.
  - credentialed (causal) retrieval: match on the CAUSAL signature (the back-door-
    adjusted effect carried in the memory's credential) -> returns the causally-
    nearest memory.
As confounding strength grows, association decouples from causation, so flat
retrieval systematically transfers the wrong effect while credentialed retrieval
stays correct. This is the cross-pillar point: the causal credential (pillar 1)
makes memory (pillar 2) verifiably correct for causal transfer.

Pure simulation, no API. Run: python experiments/memory_causal_cliff/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
M = 200            # memories in the bank
TRIALS = 400
SEEDS = range(30)
CONF = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]   # confounding shift std (association vs causation gap)


def trial(rng, conf_std):
    # bank: each memory has a true causal effect c and a confounded association a
    c = rng.uniform(-2, 2, M)
    a = c + rng.normal(0, conf_std, M)
    # query: a fresh intervention with true causal effect cq; we observe its
    # association aq (confounded) and, via its credential, its adjusted effect chat
    cq = rng.uniform(-2, 2)
    aq = cq + rng.normal(0, conf_std)
    chat = cq + rng.normal(0, 0.10)        # credential gives a slightly noisy de-confounded estimate
    # flat retrieval: nearest by association -> transfer its causal effect
    i_flat = int(np.argmin(np.abs(a - aq)))
    err_flat = abs(c[i_flat] - cq)
    # credentialed retrieval: nearest by causal signature
    i_cred = int(np.argmin(np.abs(c - chat)))
    err_cred = abs(c[i_cred] - cq)
    return err_flat, err_cred


def main():
    rows = {}
    for cs in CONF:
        ef, ec = [], []
        for s in SEEDS:
            rng = np.random.default_rng(7000 + s)
            for _ in range(TRIALS):
                a, b = trial(rng, cs)
                ef.append(a); ec.append(b)
        rows[cs] = {"flat_mae": round(float(np.mean(ef)), 4), "cred_mae": round(float(np.mean(ec)), 4)}
    print("memory retrieval for causal transfer — error in transferred causal effect\n")
    print(f"{'confounding σ':>14} | {'flat (cosine) MAE':>18} | {'credentialed MAE':>16} | {'flat/cred':>9}")
    for cs in CONF:
        f, c = rows[cs]["flat_mae"], rows[cs]["cred_mae"]
        print(f"{cs:>14.2f} | {f:>18.4f} | {c:>16.4f} | {f/max(c,1e-9):>8.1f}x")
    (HERE / "results.json").write_text(json.dumps({"M": M, "rows": rows}, indent=2))
    print("\nReading: at zero confounding, association = causation -> flat retrieval is fine")
    print("(matches WEAK-02: no average advantage). As confounding grows, association")
    print("decouples from causation -> flat retrieval transfers the WRONG effect, while")
    print("credentialed (causal-signature) retrieval stays correct. The causal credential")
    print("makes memory verifiably correct for causal decisions — the cross-pillar unification.")


if __name__ == "__main__":
    main()
