"""B3↔B1 integration · the O(N) SSM encoder feeds a VERIFIABLE causal latent.

Connects the two B-line backbones: continuous-stream perception (B3, O(N) SSM) and the
verifiable causal latent (B1, do with adjustment). Each unit has a latent confounder U_i
observed only as a NOISY SEQUENCE of T measurements s_i(t)=U_i+ε(t). The SSM encoder
(linear, O(N) in T) integrates the sequence into a state summary that DENOISES U_i; that
summary is the adjustment representation for a verifiable do().

Thesis: efficient O(N) perception and verifiable causal inference compose — and longer
sequences let the SSM's temporal integration recover the confounder better, so the do()
residual shrinks with T (and stays recomputable). Honest scope: fixed linear SSM + logistic
adjustment, numpy CPU; a feasibility bridge, not a tuned system.

Run:  .venv/bin/python experiments/bline_ssm_causal/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from theone.layer1_perception.ssm_encoder import SSMEncoder

HERE = Path(__file__).parent
A, B, C = 1.2, 1.5, 1.8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def true_do():
    v, _ = quad(lambda u: sigmoid(B + C * u) * norm.pdf(u), -np.inf, np.inf)
    return float(v)


def _fit_logistic(X, y, iters=200):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = sigmoid(X @ w); Wd = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve((X * Wd[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - mu))
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def ssm_summary(enc, seqs):
    """Encode each unit's proxy sequence with the O(N) SSM; summary = mean hidden state."""
    feats = []
    for s in seqs:
        H = enc.encode(s.reshape(-1, 1))      # (T, hidden) — O(T) scan
        feats.append(H.mean(axis=0))          # temporal integration -> denoised confounder rep
    return np.array(feats)


def do_via_summary(feat, x, y):
    design = np.column_stack([np.ones(len(y)), x, feat])
    w = _fit_logistic(design, y)
    do_d = np.column_stack([np.ones(len(y)), np.ones(len(y)), feat])
    return float(np.mean(sigmoid(do_d @ w)))


def main():
    truth = true_do()
    print("=== B3↔B1 · O(N) SSM encoder feeds a verifiable causal latent ===\n")
    print(f"truth do(X=1) = {truth:.4f}   (confounder seen only as a noisy T-step sequence)\n")
    enc = SSMEncoder(input_dim=1, hidden_dim=24, seed=0)
    print(f"SSM spectral radius = {enc.spectral_radius:.3f} (stable)\n")

    n = 1500
    rng = np.random.default_rng(0)
    u = rng.standard_normal(n)
    x = (rng.random(n) < sigmoid(A * u)).astype(float)
    y = (rng.random(n) < sigmoid(B * x + C * u)).astype(float)

    print(f"{'T (seq len)':>11} {'do via SSM latent':>18} {'residual':>10} {'recompute gap':>14}")
    rows = []
    for T in (1, 4, 16, 64):
        seqs = u[:, None] + rng.normal(0, 1.0, (n, T))     # noisy sequence carrying U
        feat = ssm_summary(enc, seqs)
        d = do_via_summary(feat, x, y)
        half = n // 2
        gap = abs(do_via_summary(feat[:half], x[:half], y[:half])
                  - do_via_summary(feat[half:], x[half:], y[half:]))
        rows.append((T, d, abs(d - truth), gap))
        print(f"{T:>11} {d:>18.4f} {abs(d-truth):>10.4f} {gap:>14.4f}")

    shrinks = rows[-1][2] < rows[0][2]
    recomputable = all(r[3] < 0.05 for r in rows)
    final_ok = rows[-1][2] < 0.05
    gate = shrinks and recomputable and final_ok
    print("\nB3↔B1 gate:")
    print(f"  longer sequence -> SSM denoises -> residual shrinks ... {'PASS' if shrinks else 'FAIL'} "
          f"({rows[0][2]:.3f} -> {rows[-1][2]:.3f})")
    print(f"  do via SSM latent is split-half recomputable ......... {'PASS' if recomputable else 'FAIL'}")
    print(f"  final residual < 0.05 (verifiable) ................... {'PASS' if final_ok else 'FAIL'}")
    print(f"\n  >>> {'PASS — O(N) perception and verifiable causal inference compose' if gate else 'CHECK'}")
    print("\nMeaning: the efficient O(N) SSM front-end and the verifiable do() back-end work as one;")
    print("temporal integration recovers the latent confounder, and the do() stays recomputable.")
    (HERE / "results.json").write_text(json.dumps(
        {"truth": round(truth, 6),
         "rows": [{"T": r[0], "do": round(r[1], 4), "residual": round(r[2], 4),
                   "recompute_gap": round(r[3], 4)} for r in rows], "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
