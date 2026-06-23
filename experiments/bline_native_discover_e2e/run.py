"""B5 · learned structure DISCOVERY -> extrapolating native do() -> engine audit (propose-and-verify).

The honest caveat threading NOTE-135/136/139/140 is "structure assumed known". This closes it: from
RAW observational samples (known causal ORDER — standard — but UNKNOWN edges), DISCOVER the DAG via
a BIC-scored greedy parent search, then run the size-extrapolating native-do GNN (NOTE-139/140) on
the DISCOVERED graph, with the exact engine AUDITING. Propose-and-verify: when discovery is unstable
(an edge sits near the BIC margin), the system ABSTAINS rather than commit to a shaky structure.

Measured: edge-recovery F1 (discovered vs true), do() error vs ground truth (should converge with n),
and the engine audit drift (native do vs exact-on-discovered). The recomputable engine stays oracle.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_native_discover_e2e/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "bline_native_do_gnn"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gnn_mod", ROOT / "experiments/bline_native_do_gnn/run.py")
g = _ilu.module_from_spec(_spec); sys.modules["gnn_mod"] = g; _spec.loader.exec_module(g)
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = g.DEVICE


def ancestral(K, parents, bias, w, n, rng):
    S = np.zeros((n, K), np.int8)
    for i in range(K):
        z = np.full(n, bias[i])
        for j in range(i):
            if parents[i][j]: z += w[i][j] * (2 * S[:, j] - 1)
        S[:, i] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.int8)
    return S


def fit_logreg(Xp, y):
    """logistic MLE -> (weights, bias, loglik)."""
    n = len(y); wv = np.zeros(Xp.shape[1]); b = 0.0
    for _ in range(800):
        p = 1 / (1 + np.exp(-(Xp @ wv + b))); gr = p - y
        wv -= 0.5 * (Xp.T @ gr) / n; b -= 0.5 * gr.mean()
    p = np.clip(1 / (1 + np.exp(-(Xp @ wv + b))), 1e-9, 1 - 1e-9)
    ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return wv, b, ll


def discover(S, margin_out=None):
    """known order 0..K-1; for each node greedily add the parent that most improves BIC.
    Records the BIC margin of the last accepted/first rejected edge for an abstain signal."""
    n, K = S.shape
    parents = [np.zeros(i, bool) for i in range(K)]
    bias = np.zeros(K); w = [np.zeros(i) for i in range(K)]
    min_margin = 1e9
    for i in range(K):
        y = S[:, i].astype(float)
        chosen = []
        # baseline: no parents
        m = np.clip(y.mean(), 1e-9, 1 - 1e-9); ll0 = float(np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)))
        best_bic = -2 * ll0 + 0 * np.log(n)
        cand = list(range(i))
        while cand:
            scores = []
            for j in cand:
                S_try = chosen + [j]
                Xp = (2 * S[:, S_try] - 1).astype(float)
                _, _, ll = fit_logreg(Xp, y)
                bic = -2 * ll + len(S_try) * np.log(n)
                scores.append((bic, j))
            scores.sort()
            bic_best, j_best = scores[0]
            margin = best_bic - bic_best                      # improvement from adding j_best
            if margin > 0:
                chosen.append(j_best); cand.remove(j_best); best_bic = bic_best
                if margin < min_margin: min_margin = margin
            else:
                if abs(margin) < min_margin: min_margin = abs(margin)
                break
        chosen.sort()
        for j in chosen: parents[i][j] = True
        if chosen:
            Xp = (2 * S[:, chosen] - 1).astype(float); wv, b, _ = fit_logreg(Xp, y)
            for idx, j in enumerate(chosen): w[i][j] = wv[idx]
            bias[i] = b
        else:
            bias[i] = np.log(m / (1 - m))
    if margin_out is not None: margin_out.append(min_margin)
    return parents, bias, w


def f1_edges(true_p, disc_p, K):
    tp = fp = fn = 0
    for i in range(K):
        for j in range(i):
            t = bool(true_p[i][j]); d = bool(disc_p[i][j])
            tp += t and d; fp += d and not t; fn += t and not d
    prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def train_gnn():
    Gtr = g.make([4, 5], 4000 if FAST else 14000, 0)
    net = g.GNNdo().to(DEVICE); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    for ep in range(6 if FAST else 16):
        net.train(); order = np.random.default_rng(100 + ep).permutation(len(Gtr))
        for c, i in enumerate(order):
            nf, edges, X, Y, d, K, _ = Gtr[i]
            t = torch.tensor(nf, device=DEVICE)
            loss = (net(t, edges, X, Y) - torch.tensor(d, dtype=torch.float32, device=DEVICE)) ** 2
            loss.backward()
            if (c + 1) % 32 == 0: opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    net.eval(); return net


def main():
    torch.manual_seed(0)
    print("=== B5 · learned DISCOVERY -> extrapolating native do -> engine audit (propose & verify) ===\n")
    net = train_gnn()
    rng = np.random.default_rng(7)
    K = 5
    for n_obs in [400, 2000, 8000]:
        f1s = []; do_err = []; audit = []; abst = 0; spoke = 0
        margins = []
        N = 60 if FAST else 150
        for _ in range(N):
            parents, bias, w, X, Y = g.sample(K, rng)
            true_do = g.do_exact(K, X, Y, parents, bias, w)
            S = ancestral(K, parents, bias, w, n_obs, rng)
            mo = []
            dp, db, dw = discover(S, mo)
            f1s.append(f1_edges(parents, dp, K))
            # propose-and-verify: abstain if the discovery margin is shaky
            if mo[0] < 3.0:
                abst += 1; continue
            nf, edges, Xi, Yi = g.to_graph(K, X, Y, dp, db, dw)
            with torch.no_grad():
                native = float(net(torch.tensor(nf, device=DEVICE), edges, Xi, Yi).cpu())
            eng = g.do_exact(K, X, Y, dp, db, dw)              # engine audit on discovered structure
            do_err.append(abs(native - true_do)); audit.append(abs(native - eng)); spoke += 1
        print(f"  n_obs={n_obs:>5}:  edge-F1={np.mean(f1s):.3f}  spoke={spoke}/{N} (abstained {abst})  "
              f"do-MAE={np.mean(do_err):.4f}  audit-drift={np.mean(audit):.4f}")
        last = (np.mean(f1s), np.mean(do_err), np.mean(audit))

    f1_final, do_final, aud_final = last
    g1 = f1_final > 0.85                                       # discovery recovers the true edges at large n
    g2 = do_final < 0.07                                       # do from DISCOVERED structure stays close to truth
    g3 = aud_final < 0.07                                      # native do matches engine on discovered structure
    allok = g1 and g2 and g3
    print("\ndiscover->do gate (largest n):")
    print(f"  [{'PASS' if g1 else 'FAIL'}] learned structure discovery recovers true edges (F1>0.85)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] native do() on DISCOVERED structure stays close to ground-truth do (<0.07)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] engine audits the native do on the discovered structure (drift<0.07)")
    msg = ("PASS — structure is now DISCOVERED from raw data (not given): discover -> extrapolating "
           "native do -> engine-audited, with propose-and-verify abstention. The last "
           "structure-assumed-known caveat is closed (given causal order).") if allok else "CHECK"
    print(f"\n  >>> {msg}")
    print("\nHonest: causal ORDER assumed known (standard; edges discovered); latent confounding still")
    print("bounded by identifiability (NOTE-076); the exact engine remains the recomputable oracle.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
