"""B1 REAL-SCALE · the core kill-gate with a real PyTorch neural network on the GPU.

The numpy toy (probe 6/7) passed. The real test: does the verifiable-latent thesis survive
a REAL neural network at scale? A multi-layer PyTorch encoder learns a continuous causal
latent from many NONLINEAR noisy proxies of MULTIPLE latent confounders; we check that do()
through the learned latent stays (a) accurate vs truth, (b) split-half recomputable, (c)
convergent with data, and (d) carries a truth-free abstain signal. Runs on Apple-Silicon
GPU (MPS) if available.

If this holds at real scale -> B1's kill-gate is genuinely de-risked (not just a toy).
If it breaks -> we learn exactly where, and pivot per docs/THE_ONE_BLINE_PLAN.md.

Run:  .venv/bin/python experiments/bline_real_b1/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad
from scipy.stats import norm

HERE = Path(__file__).parent
DEV = torch.device("mps" if torch.backends.mps.is_available()
                   else ("cuda" if torch.cuda.is_available() else "cpu"))
KU, P_EACH = 4, 3          # 4 latent confounders, 3 nonlinear proxies each
B_COEF, C_COEF = 1.5, 0.9  # X->Y, each U->Y


def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def true_do_x1():
    s = C_COEF * np.sqrt(KU)                       # sum of KU iid N(0,1) ~ N(0, KU)
    v, _ = quad(lambda z: 1/(1+np.exp(-(B_COEF + z))) * norm.pdf(z, scale=s), -np.inf, np.inf)
    return float(v)


def gen(n, sigma, seed):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n, KU))
    cols = []
    for j in range(KU):
        for _ in range(P_EACH):
            cols.append(np.tanh(rng.uniform(1.0, 2.2) * U[:, j]) + rng.normal(0, sigma, n))
    P = np.column_stack(cols).astype(np.float32)
    usum = U.sum(1)
    x = (rng.random(n) < sigmoid(1.0 * usum)).astype(np.float32)
    y = (rng.random(n) < sigmoid(B_COEF * x + C_COEF * usum)).astype(np.float32)
    return P, x, y


class Encoder(nn.Module):
    def __init__(self, k, latent=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(k, 128), nn.SiLU(),
                                 nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, latent))
        self.head = nn.Linear(latent + 1, 1)       # X appended -> additive treatment effect
    def forward(self, P, x):
        h = self.enc(P)
        return self.head(torch.cat([h, x[:, None]], 1)).squeeze(1)


def fit_and_do(P, x, y, epochs=120, seed=0):
    torch.manual_seed(seed)
    Pt = torch.tensor(P, device=DEV); xt = torch.tensor(x, device=DEV); yt = torch.tensor(y, device=DEV)
    net = Encoder(P.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    nbatch = max(1, len(y) // 4096)
    for _ in range(epochs):
        perm = torch.randperm(len(y), device=DEV)
        for b in range(nbatch):
            idx = perm[b*4096:(b+1)*4096]
            opt.zero_grad(); loss = lossf(net(Pt[idx], xt[idx]), yt[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        do1 = torch.sigmoid(net(Pt, torch.ones_like(xt))).mean().item()
    return float(do1)


def proxy_reliability(P):
    """Truth-free signal: fraction of proxy variance in the top-KU principal directions.
    Clean proxies concentrate variance in the KU confounder directions (high); noisy
    proxies spread variance across all dims (low). Works for multiple confounders, where
    cross-confounder proxy pairs are uncorrelated and a global mean-correlation fails."""
    Pc = P - P.mean(0)
    s = np.linalg.svd(Pc, compute_uv=False)
    var = s ** 2
    return float(var[:KU].sum() / var.sum())


def main():
    print(f"=== B1 REAL-SCALE · PyTorch causal latent on {str(DEV).upper()} ===\n")
    truth = true_do_x1()
    print(f"device = {DEV} · truth do(X=1) = {truth:.4f} · {KU} latent confounders, "
          f"{KU*P_EACH} nonlinear proxies\n")

    # convergence: residual shrinks as data grows
    print(f"{'n':>8} {'do(learned)':>12} {'residual':>10} {'recompute gap':>14}")
    rows = []
    for n in (5000, 20000, 80000):
        P, x, y = gen(n, 0.4, seed=0)
        do_full = fit_and_do(P, x, y, seed=1)
        h = n // 2
        gap = abs(fit_and_do(P[:h], x[:h], y[:h], seed=1) - fit_and_do(P[h:], x[h:], y[h:], seed=1))
        rows.append((n, do_full, abs(do_full - truth), gap))
        print(f"{n:>8} {do_full:>12.4f} {abs(do_full-truth):>10.4f} {gap:>14.4f}")

    # abstain signal: noisy proxies -> reliability drops
    Pc, _, _ = gen(20000, 0.4, 9); Pn, _, _ = gen(20000, 1.8, 9)
    rel_c, rel_n = proxy_reliability(Pc), proxy_reliability(Pn)
    print(f"\nabstain signal (proxy reliability): clean={rel_c:.3f}  noisy={rel_n:.3f}  "
          f"({'separates' if rel_c > rel_n + 0.2 else 'weak'})")

    converges = rows[-1][2] < rows[0][2]
    accurate = rows[-1][2] < 0.05
    recomputable = rows[-1][3] < 0.03
    abstains = rel_c > rel_n + 0.2
    gate = converges and accurate and recomputable and abstains
    print("\nB1 REAL-SCALE gate:")
    print(f"  real NN: do converges with data ............ {'PASS' if converges else 'FAIL'} "
          f"({rows[0][2]:.3f}->{rows[-1][2]:.3f})")
    print(f"  accurate vs truth (residual < 0.05) ........ {'PASS' if accurate else 'FAIL'}")
    print(f"  split-half recomputable (gap < 0.03) ....... {'PASS' if recomputable else 'FAIL'}")
    print(f"  truth-free abstain signal separates ........ {'PASS' if abstains else 'FAIL'}")
    print(f"\n  >>> {'PASS — verifiable causal latent SURVIVES a real NN at scale (B1 de-risked)' if gate else 'CHECK — see which property broke; pivot per plan'}")
    print("\nHonest scope: real PyTorch encoder on GPU, large synthetic multi-confounder data.")
    print("Next: public causal benchmarks (IHDP/ACIC) for external validity; then RunPod scale.")
    (HERE / "results.json").write_text(json.dumps(
        {"device": str(DEV), "truth": round(truth, 6),
         "rows": [{"n": r[0], "do": round(r[1], 4), "residual": round(r[2], 4),
                   "gap": round(r[3], 4)} for r in rows],
         "rel_clean": round(rel_c, 3), "rel_noisy": round(rel_n, 3), "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
