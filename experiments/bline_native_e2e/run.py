"""B5 capstone · END-TO-END native verifiable cognition FROM RAW DATA.

NOTE-135 showed one net computes do() for ARBITRARY structures — but amortized from GIVEN CPTs.
The complete form must run from RAW OBSERVATIONS. This closes that gap end-to-end:

  raw observational samples (unknown SCM, arbitrary structure)
    -> estimate each node's CPT from data (per-node logistic MLE)
    -> structure-general native-do net (NOTE-135) computes do(X=1) from the ESTIMATED params
    -> exact enumeration engine AUDITS the native estimate (recompute on the same estimated params)
    -> credential: ANSWER if audit drift is low; the estimate converges to ground-truth do as n grows

This is native (a learned net does the inference), structure-general (any DAG), verifiable (the
engine recomputes), and honest (it reports its distance to the recomputable oracle, and estimation
error shrinks with sample size). Structure is assumed known here; PARAMETERS come from raw data —
the discovery leg is covered separately (bline_native_discover_do).

Run:  .venv/bin/python experiments/bline_native_e2e/run.py   (THEONE_FAST=1 for a smoke)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "bline_native_do_varstruct"))
import run as vs                                              # reuse the structure-general do machinery
FAST = os.environ.get("THEONE_FAST") == "1"
K = vs.K
DEVICE = vs.DEVICE


def ancestral_sample(parents, bias, w, n, rng):
    """draw n observational samples from the SCM by ancestral sampling."""
    S = np.zeros((n, K), dtype=np.int8)
    for i in range(K):
        z = np.full(n, bias[i])
        for j in range(i):
            if parents[i][j]:
                z += w[i][j] * (2 * S[:, j] - 1)
        p = 1.0 / (1.0 + np.exp(-z))
        S[:, i] = (rng.random(n) < p).astype(np.int8)
    return S


def estimate_cpts(S, parents):
    """per-node logistic MLE of P(node=1 | parents) from samples -> (bias_hat, w_hat) per node."""
    n = len(S)
    bias_hat = np.zeros(K); w_hat = [np.zeros(i) for i in range(K)]
    for i in range(K):
        pa = [j for j in range(i) if parents[i][j]]
        y = S[:, i].astype(np.float64)
        if not pa:
            m = np.clip(y.mean(), 1e-3, 1 - 1e-3); bias_hat[i] = np.log(m / (1 - m)); continue
        Xp = (2 * S[:, pa] - 1).astype(np.float64)           # parents as +-1
        wv = np.zeros(len(pa)); b = 0.0
        for _ in range(1500):                                 # logistic MLE (well-converged, no shrinkage)
            p = 1.0 / (1.0 + np.exp(-(Xp @ wv + b)))
            g = p - y
            wv -= 0.6 * (Xp.T @ g) / n; b -= 0.6 * g.mean()
        for idx, j in enumerate(pa):
            w_hat[i][j] = wv[idx]
        bias_hat[i] = b
    return bias_hat, w_hat


def main():
    torch.manual_seed(0)
    print("=== B5 capstone · END-TO-END native verifiable cognition FROM RAW DATA ===\n")
    # 1) train the structure-general native-do net (NOTE-135 machinery)
    NTR = 14000 if FAST else 40000
    Xtr, ytr, _ = vs.make(NTR, 0, with_obs=False)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor(((Xtr - mu) / sd), device=DEVICE); yt = torch.tensor(ytr, device=DEVICE)
    net = vs.Net(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    for ep in range(140 if FAST else 220):
        net.train(); perm = torch.randperm(len(xt), device=DEVICE)
        for i in range(0, len(xt), 1024):
            idx = perm[i:i + 1024]
            loss = torch.nn.functional.mse_loss(net(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    print(f"  trained structure-general native-do net (train mse {loss.item():.5f})\n")

    # 2) end-to-end: raw samples -> estimate CPTs -> native do -> engine audit, across sample sizes
    # decompose the two error sources honestly:
    #   estimation error = |engine_do(estimated params) - true do|   (pure sampling; SHRINKS with n)
    #   net fidelity     = |native_do(est) - engine_do(est)|         (the net's approx error; audit drift)
    #   e2e              = |native_do(est) - true do|                (bounded by their sum)
    rng = np.random.default_rng(7)
    n_eval = 140 if FAST else 800
    est_by_n = {}; aud_last = None
    for n_obs in [200, 1000, 5000]:
        e2e_err = []; audit_drift = []; est_err = []; answered = 0
        for _ in range(n_eval):
            parents, bias, w, X, Y = vs.sample_dag(rng)
            true_do = vs.do_exact(X, Y, parents, bias, w)             # ground truth (true params)
            S = ancestral_sample(parents, bias, w, n_obs, rng)
            bh, wh = estimate_cpts(S, parents)                        # estimate params from RAW data
            feat = vs.featurize(X, Y, parents, bh, wh)
            xf = torch.tensor(((np.array(feat, np.float32) - mu) / sd)[None], device=DEVICE)
            with torch.no_grad():
                native = float(net(xf).cpu().numpy()[0])             # native do() on estimated params
            engine_est = vs.do_exact(X, Y, parents, bh, wh)          # engine AUDIT on same estimated params
            e2e_err.append(abs(native - true_do))
            est_err.append(abs(engine_est - true_do))                # isolates SAMPLING convergence
            drift = abs(native - engine_est); audit_drift.append(drift)
            if drift < (0.05 if FAST else 0.02):                      # credential: recomputable -> ANSWER
                answered += 1
        est = float(np.mean(est_err)); aud = float(np.mean(audit_drift)); e2e = float(np.mean(e2e_err))
        est_by_n[n_obs] = est; aud_last = aud
        print(f"  n_obs={n_obs:>5}:  estimation→truth MAE={est:.4f} (sampling)   "
              f"net-fidelity(audit)={aud:.4f}   e2e={e2e:.4f}   answered={100*answered/n_eval:.0f}%")

    net_thr = 0.065 if FAST else 0.02                         # net floor is scale-dependent (NOTE-135: 0.007 full)
    g1 = est_by_n[5000] < est_by_n[200] and est_by_n[5000] < 0.02   # estimation converges with data
    g2 = aud_last < net_thr                                   # native-do matches engine on estimated params
    allok = g1 and g2
    print("\nend-to-end native-cognition gate (decomposed):")
    print(f"  [{'PASS' if g1 else 'FAIL'}] estimation from RAW data converges to truth as n grows "
          f"({est_by_n[200]:.3f}→{est_by_n[5000]:.3f}, <0.02 @ n=5000)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] native-do matches the exact engine on estimated params (audit < {net_thr})")
    print(f"\n  >>> {'PASS — END-TO-END native verifiable cognition from raw data: estimate(convergent) -> native do(engine-tight) -> audited credential. e2e error = net floor (NOTE-135: 0.007 full / 0.035 smoke) + sampling, both reported' if allok else 'CHECK'}")
    print("\nHonest: structure assumed known (discovery leg = bline_native_discover_do); the net's own")
    print("approximation error (the audit drift) is the e2e floor and is scale-dependent — full-scale")
    print("NOTE-135 net = 0.007; the enumeration engine stays the recomputable oracle throughout.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
