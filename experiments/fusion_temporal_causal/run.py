"""Fusion deepening⑩ · temporal causal direction on continuous streams (L1->L2 for time).

A two-stream system where A drives B with a one-step lag (B[t] depends on A[t-1], not
vice versa). The verifiable lagged-causality test should recover A->B and, on a symmetric
control (two independent streams), ABSTAIN. This closes the perception->causal-direction
seam for time series with the project's abstain discipline.

Run:  .venv/bin/python experiments/fusion_temporal_causal/run.py
"""
from __future__ import annotations
import numpy as np

from theone.layer2_world_model.temporal_causal import temporal_direction


def var_a_drives_b(n, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros(n); b = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.5 * a[t - 1] + rng.normal(0, 1)
        b[t] = 0.4 * b[t - 1] + 0.6 * a[t - 1] + rng.normal(0, 1)   # A[t-1] -> B[t]
    return a, b


def independent(n, seed):
    rng = np.random.default_rng(seed)
    a = np.zeros(n); b = np.zeros(n)
    for t in range(1, n):
        a[t] = 0.5 * a[t - 1] + rng.normal(0, 1)
        b[t] = 0.5 * b[t - 1] + rng.normal(0, 1)                    # no cross-coupling
    return a, b


def main():
    print("=== Fusion deepening⑩: temporal causal direction (L1->L2 for streams) ===\n")
    ok = True

    a, b = var_a_drives_b(4000, 0)
    r = temporal_direction(a, b)
    print(f"A drives B (lag 1): verdict={r['verdict']}  "
          f"p(a->b)={r['p_ab']:.1e}  p(b->a)={r['p_ba']:.2f}  "
          f"R2gain a->b={r['r2_gain_ab']} vs b->a={r['r2_gain_ba']}")
    ok &= r["verdict"] == "a->b"

    a2, b2 = independent(4000, 1)
    r2 = temporal_direction(a2, b2)
    print(f"independent streams: verdict={r2['verdict']}  "
          f"p(a->b)={r2['p_ab']:.2f}  p(b->a)={r2['p_ba']:.2f}")
    ok &= r2["verdict"] == "abstain"

    print("\nReading: the directed claim A->B is made only on a clear asymmetry (A's past")
    print("improves B's prediction AND B's past does not improve A's); independent streams give")
    print("no asymmetry -> ABSTAIN. Causal direction, not symmetric correlation, with the F-test")
    print("as the recomputable credential — the founding asymmetry, now on continuous time series.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
