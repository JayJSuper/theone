"""B3 · SSM O(N) backbone — empirically confirm linear-time scaling vs O(N^2) attention,
and verifiable reconstruction quality. CPU.

Two measurements:
  1. TIME vs sequence length N: the SSM encoder (sequential scan) should scale ~O(N);
     a self-attention-style op (N×N similarity) scales ~O(N^2). We fit log-log slopes —
     SSM slope ≈ 1, attention slope ≈ 2 — so the architectural claim is measured, not asserted.
  2. RECONSTRUCTION: the SSM latent must genuinely encode the signal (D=0, so reconstruction
     flows through the state), MSE < 1e-3 on smooth signals.

This is the O(N) backbone the B-line native engine needs (per plan B3). Honest scope: a
self-implemented linear SSM on CPU; not a tuned Mamba kernel, but the SCALING is the point.

Run:  .venv/bin/python experiments/bline_ssm_scaling/run.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from theone.layer1_perception.ssm_encoder import SSMEncoder

HERE = Path(__file__).parent


def _attention_forward(x):
    """O(N^2) self-attention-style op (the baseline SSM avoids)."""
    n = len(x)
    v = x.reshape(n, -1)
    scores = v @ v.T                      # N×N  -> O(N^2)
    scores -= scores.max(axis=1, keepdims=True)
    w = np.exp(scores); w /= w.sum(axis=1, keepdims=True)
    return w @ v


def _signal(n, seed=0):
    t = np.linspace(0, 30, n)
    rng = np.random.default_rng(seed)
    return (np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 1.3 * t)
            + 0.05 * rng.standard_normal(n))


def _loglog_slope(ns, ts):
    a, _ = np.polyfit(np.log(ns), np.log(ts), 1)
    return float(a)


def _time(fn, x, repeat=3):
    fn(x)  # warmup
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(x); best = min(best, time.perf_counter() - t0)
    return best


def main():
    print("=== B3 · SSM O(N) backbone: measured scaling + reconstruction ===\n")
    enc = SSMEncoder(input_dim=1, hidden_dim=64, seed=0)
    print(f"spectral radius = {enc.spectral_radius:.3f}  (<1 required for stability)")

    Ns = [500, 1000, 2000, 4000, 8000]
    ssm_t, att_t = [], []
    print(f"\n{'N':>7} {'SSM time(ms)':>14} {'attention(ms)':>14}")
    for n in Ns:
        x = _signal(n).reshape(-1, 1)
        st = _time(lambda s: enc.encode(s), x)
        at = _time(lambda s: _attention_forward(s), x)
        ssm_t.append(st); att_t.append(at)
        print(f"{n:>7} {st*1e3:>14.2f} {at*1e3:>14.2f}")

    ssm_slope = _loglog_slope(Ns, ssm_t)
    att_slope = _loglog_slope(Ns, att_t)
    print(f"\nlog-log slope:  SSM = {ssm_slope:.2f} (≈1 ⇒ O(N))   attention = {att_slope:.2f} (≈2 ⇒ O(N²))")

    # reconstruction quality
    x = _signal(4000).reshape(-1, 1)
    H = enc.encode(x); enc.fit_decoder(x, H)
    mse = enc.reconstruction_mse(x, H)
    print(f"reconstruction MSE (4000-step signal) = {mse:.2e}  (target < 1e-3)")

    linear = ssm_slope < 1.3
    sub_quadratic = ssm_slope < att_slope - 0.5
    recon_ok = mse < 1e-3
    gate = linear and sub_quadratic and recon_ok
    print("\nB3 gate:")
    print(f"  SSM scales ~linearly (slope < 1.3) ............ {'PASS' if linear else 'FAIL'}")
    print(f"  clearly sub-quadratic vs attention ............ {'PASS' if sub_quadratic else 'FAIL'}")
    print(f"  reconstruction MSE < 1e-3 ..................... {'PASS' if recon_ok else 'FAIL'}")
    print(f"\n  >>> B3 backbone: {'PASS — measured O(N) + verifiable reconstruction' if gate else 'CHECK'}")
    (HERE / "results.json").write_text(json.dumps(
        {"Ns": Ns, "ssm_ms": [round(t*1e3, 3) for t in ssm_t],
         "att_ms": [round(t*1e3, 3) for t in att_t], "ssm_slope": round(ssm_slope, 3),
         "att_slope": round(att_slope, 3), "reconstruction_mse": mse, "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
