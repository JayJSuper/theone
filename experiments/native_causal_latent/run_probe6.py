"""Native causal latent — probe 6: the B1 kill-gate, minimal CPU version.

THE HARD QUESTION (B1): can a *neural-network-learned* latent preserve a VERIFIABLE do()?
Probe 4 learned only a LINEAR adjustment. Probe 6 makes the confounder observable ONLY
through NONLINEAR noisy proxies, so a linear adjustment is biased — and asks whether a
small learned MLP encoder recovers a latent in which do() is (a) recomputable, (b)
convergent to truth, and (c) honestly abstaining when the latent is insufficient.

Setup: latent U~N(0,1) (never observed). Proxies P_j = tanh(a_j·U) + noise — monotonic
but NONLINEAR, so U cannot be recovered by a linear map. X~Bern(σ(1.2U)),
Y~Bern(σ(1.5X + 1.8U)). do(X=1) = E_U[σ(1.5+1.8U)] (known truth).

We compare a LINEAR backdoor adjustment vs a learned 1-hidden-layer MLP encoder (pure
numpy, gradient descent — no torch, CPU). The MLP must beat linear AND stay verifiable.

Gate (pass = B1 minimal thesis holds): MLP residual-vs-truth < linear residual, the MLP
do() is split-half recomputable (small gap), and a truth-free incompleteness signal rises
when proxies are degraded (the abstain handle).

Run:  .venv/bin/python experiments/native_causal_latent/run_probe6.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

HERE = Path(__file__).parent
A, B, C = 1.2, 1.5, 1.8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def true_do_x1():
    val, _ = quad(lambda u: sigmoid(B + C * u) * norm.pdf(u), -np.inf, np.inf)
    return float(val)


def gen(n, p, sigma, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    a = rng.uniform(1.0, 2.2, p)                      # nonlinear proxy gains
    P = np.tanh(u[:, None] * a[None, :]) + rng.normal(0, sigma, (n, p))
    x = (rng.random(n) < sigmoid(A * u)).astype(float)
    y = (rng.random(n) < sigmoid(B * x + C * u)).astype(float)
    return P, x, y


# ---- linear backdoor adjustment (probe-4 style) ------------------------------
def _fit_logistic(X, y, iters=300):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = sigmoid(X @ w)
        Wd = np.clip(mu * (1 - mu), 1e-9, None)
        step = np.linalg.solve((X * Wd[:, None]).T @ X + 1e-6 * np.eye(X.shape[1]), X.T @ (y - mu))
        w = w + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def linear_do(P, x, y):
    design = np.column_stack([np.ones(len(y)), x, P])
    w = _fit_logistic(design, y)
    do_design = np.column_stack([np.ones(len(y)), np.ones(len(y)), P])
    return float(np.mean(sigmoid(do_design @ w)))


# ---- learned MLP encoder + outcome head (numpy, gradient descent) ------------
class MLPDo:
    """hidden = tanh(P W1 + b1);  logit = hidden·v + u·X + b2. X enters additively at the
    output so `hidden(P)` is the learned confounder representation (the latent)."""
    def __init__(self, k, H=16, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(k)
        self.W1 = rng.normal(0, s, (k, H)); self.b1 = np.zeros(H)
        self.v = rng.normal(0, 1.0 / np.sqrt(H), H); self.u = 0.0; self.b2 = 0.0

    def _hidden(self, P):
        return np.tanh(P @ self.W1 + self.b1)

    def _logit(self, P, x, hid=None):
        hid = self._hidden(P) if hid is None else hid
        return hid @ self.v + self.u * x + self.b2

    def fit(self, P, x, y, lr=0.2, epochs=400):
        n = len(y)
        for _ in range(epochs):
            hid = self._hidden(P)
            p = sigmoid(self._logit(P, x, hid))
            d = (p - y) / n                                   # (n,)
            dv = hid.T @ d; du = float(d @ x); db2 = float(d.sum())
            dh = d[:, None] * self.v[None, :]
            dpre = dh * (1 - hid ** 2)
            dW1 = P.T @ dpre; db1 = dpre.sum(0)
            self.v -= lr * dv; self.u -= lr * du; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1
        return self

    def do_x1(self, P):
        return float(np.mean(sigmoid(self._logit(P, np.ones(len(P))))))


def mlp_do(P, x, y, seed=0):
    return MLPDo(P.shape[1], seed=seed).fit(P, x, y).do_x1(P)


def main():
    truth = true_do_x1()
    print("=== probe 6 · B1 kill-gate: does a LEARNED nonlinear latent keep do() verifiable? ===\n")
    print(f"truth do(X=1) = {truth:.4f}   (confounder observable only through NONLINEAR proxies)\n")

    n, p, sig = 40000, 6, 0.4
    P, x, y = gen(n, p, sig, seed=0)

    lin = linear_do(P, x, y)
    mlp = mlp_do(P, x, y, seed=1)
    # split-half recompute of the MLP do (criterion 5)
    half = n // 2
    mlp_a = mlp_do(P[:half], x[:half], y[:half], seed=1)
    mlp_b = mlp_do(P[half:], x[half:], y[half:], seed=1)
    recompute_gap = abs(mlp_a - mlp_b)

    print(f"{'method':<28}{'do(X=1)':>10}{'residual vs truth':>20}")
    print(f"{'linear adjustment':<28}{lin:>10.4f}{abs(lin-truth):>20.4f}")
    print(f"{'learned MLP latent':<28}{mlp:>10.4f}{abs(mlp-truth):>20.4f}")
    print(f"\nMLP split-half recompute gap = {recompute_gap:.4f}  (criterion 5: recomputable)")

    # abstain handle: degrade proxies (more noise) -> residual should grow, and a
    # truth-free signal (do drift across proxy subsets) should rise.
    Pn, xn, yn = gen(n, p, 1.6, seed=2)        # noisy proxies
    mlp_noisy = mlp_do(Pn, xn, yn, seed=1)
    drift_clean = abs(mlp_do(P[:, :3], x, y, seed=1) - mlp_do(P[:, 3:], x, y, seed=1))
    drift_noisy = abs(mlp_do(Pn[:, :3], xn, yn, seed=1) - mlp_do(Pn[:, 3:], xn, yn, seed=1))
    print(f"\nabstain handle (truth-free subset-drift): clean={drift_clean:.4f} -> "
          f"noisy={drift_noisy:.4f}  (rises when latent is insufficient -> abstain trigger)")

    beats_linear = abs(mlp - truth) < abs(lin - truth)
    recomputable = recompute_gap < 0.03
    drift_tracks = drift_noisy > drift_clean
    gate = beats_linear and recomputable and drift_tracks
    print("\nB1 minimal gate:")
    print(f"  learned latent beats linear adjustment ........ {'PASS' if beats_linear else 'FAIL'}")
    print(f"  learned do is split-half recomputable ......... {'PASS' if recomputable else 'FAIL'}")
    print(f"  truth-free drift rises when insufficient ...... {'PASS' if drift_tracks else 'FAIL'}")
    print(f"\n  >>> B1 minimal kill-gate: {'PASS — the learned latent keeps do() verifiable' if gate else 'FAIL — pivot per plan'}")
    print("\nHonest scope: small numpy MLP on synthetic nonlinear-proxy data, CPU. This is the")
    print("minimal feasibility probe for B1's thesis, not real-scale. Passing means the path is")
    print("worth scaling (real data + GPU); failing would tell us exactly where to pivot.")
    (HERE / "results_probe6.json").write_text(json.dumps(
        {"truth": round(truth, 6), "linear_do": round(lin, 6), "mlp_do": round(mlp, 6),
         "recompute_gap": round(recompute_gap, 6),
         "drift_clean": round(drift_clean, 6), "drift_noisy": round(drift_noisy, 6),
         "gate_pass": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
