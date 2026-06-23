"""Native perception -> native verifiable engine, end-to-end.

A latent confounder is seen only as noisy continuous SEQUENCES. The SSM perceives it into
a confounder estimate; the native engine then runs verifiable do() (replay chain + three-
zone) on the perceived confounder. Demonstrates: adjustment from PERCEPTION beats the
confounded observational estimate, and improves as sequences lengthen (SSM denoises).

Run:  .venv/bin/python experiments/native_perception/run.py
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from theone.native import NativeVerifiableEngine
from theone.native.perception import SSMPerception

A, B, C = 1.2, 1.5, 1.8


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def true_ate():
    do1, _ = quad(lambda u: sigmoid(B + C * u) * norm.pdf(u), -np.inf, np.inf)
    do0, _ = quad(lambda u: sigmoid(C * u) * norm.pdf(u), -np.inf, np.inf)
    return float(do1 - do0)


def gen(n, T, sigma, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    streams = u[:, None] + rng.normal(0, sigma, (n, T))
    x = (rng.random(n) < sigmoid(A * u)).astype(int)
    y = (rng.random(n) < sigmoid(B * x + C * u)).astype(int)
    return streams, x, y


def main():
    print("=== native perception -> native verifiable engine (continuous stream -> do) ===\n")
    truth = true_ate()
    eng = NativeVerifiableEngine()
    n = 6000
    streams_all, x, y = gen(n, 64, 1.0, seed=0)
    obs = y[x == 1].mean() - y[x == 0].mean()
    print(f"true ATE = {truth:.4f} · confounded observational ATE = {obs:.4f} "
          f"(bias {abs(obs-truth):.3f})\n")

    print(f"{'seq len T':>10} {'perceived ATE':>14} {'residual':>10} {'zone':>22} {'replay':>7}")
    rows = []
    for T in (1, 8, 64):
        streams = streams_all[:, :T]
        df = SSMPerception(seed=0).perceive_into_df(streams, x, y, n_strata=2)
        r = eng.estimate(df, confounder="U")
        rows.append((T, r.effect, abs(r.effect - truth), r.zone, r.replay_ok))
        print(f"{T:>10} {r.effect:>14.4f} {abs(r.effect-truth):>10.4f} {r.zone:>22} {str(r.replay_ok):>7}")

    helps = rows[-1][2] < abs(obs - truth)              # perception-adjusted beats observational
    improves = rows[-1][2] <= rows[0][2]                # longer streams -> better
    replay_ok = all(r[4] for r in rows)
    gate = helps and improves and replay_ok
    print("\nnative-perception gate:")
    print(f"  perception-adjusted beats confounded observational . {'PASS' if helps else 'FAIL'} "
          f"({rows[-1][2]:.3f} < {abs(obs-truth):.3f})")
    print(f"  longer sequences -> better perception/adjustment .... {'PASS' if improves else 'FAIL'} "
          f"({rows[0][2]:.3f} -> {rows[-1][2]:.3f})")
    print(f"  every estimate still replay-verifies ................ {'PASS' if replay_ok else 'FAIL'}")
    print(f"\n  >>> {'PASS — verifiable causal inference from CONTINUOUS PERCEPTION (B3 SSM + native engine)' if gate else 'CHECK'}")
    print("\nMeaning: the native engine now runs on perceived confounders from continuous streams,")
    print("not just tabular ones — SSM O(N) perception feeds replay-verified, three-zone do().")
    print("Honest: 2-stratum discretization of a continuous confounder leaves residual; finer")
    print("strata / continuous adjustment is the next refinement.")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
