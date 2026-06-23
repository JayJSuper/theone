"""B4 deepest test · SIZE-INVARIANT native do() that EXTRAPOLATES to more variables than trained on.

NOTE-135 showed one fixed-input net does do() across structures of a FIXED size (K=5). The strongest
evidence that a net internalized the do-calculus ALGORITHM (not a size-specific map) is EXTRAPOLATION:
train on small graphs, test on LARGER ones it never saw.

Architecture — a message-passing net that structurally encodes the intervention:
  do(X=1) = CUT X's incoming edges, CLAMP X=1, propagate parent->child, read Y.
Each node embeds its CPT (bias); edges carry the logistic weight w_ij; messages flow along the DAG
for R rounds (R >= max depth). Because it is a GNN (shared node/edge functions, pooled), it accepts
ANY number of variables — so we can train on K in {4,5} and TEST on K in {6,7}.

Engine oracle: exact enumeration of do(X=1) over the 2^(K-1) non-treatment assignments.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_native_do_gnn/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
Hd = 32


def sample(K, rng):
    """random DAG (topo order) over K binary vars with a forced confounder C->X,C->Y,X->Y."""
    C = int(rng.integers(0, K - 2)); X = int(rng.integers(C + 1, K - 1)); Y = int(rng.integers(X + 1, K))
    parents = [rng.random(i) < 0.45 for i in range(K)]
    parents[X][C] = True; parents[Y][C] = True; parents[Y][X] = True
    bias = rng.normal(0, 1.4, K); w = [rng.normal(0, 2.2, i) for i in range(K)]
    return parents, bias, w, X, Y


def p_one(i, a, parents, bias, w):
    z = bias[i]
    for j in range(i):
        if parents[i][j]: z += w[i][j] * (2 * a[j] - 1)
    return 1.0 / (1.0 + np.exp(-z))


def do_exact(K, X, Y, parents, bias, w, xv=1):
    others = [v for v in range(K) if v != X]; tot = 0.0
    for mask in range(1 << len(others)):
        a = [0] * K; a[X] = xv
        for b, v in enumerate(others): a[v] = (mask >> b) & 1
        pr = 1.0
        for v in others:
            pv = p_one(v, a, parents, bias, w); pr *= pv if a[v] else (1 - pv)
        if a[Y]: tot += pr
    return tot


def obs_cond(K, X, Y, parents, bias, w, xv=1):
    num = den = 0.0
    for mask in range(1 << K):
        a = [(mask >> b) & 1 for b in range(K)]; pr = 1.0
        for v in range(K):
            pv = p_one(v, a, parents, bias, w); pr *= pv if a[v] else (1 - pv)
        if a[X] == xv:
            den += pr; num += pr if a[Y] else 0
    return num / den if den else 0.0


def to_graph(K, X, Y, parents, bias, w):
    """node feats [bias, is_X, is_Y]; edges (j->i, weight). do(X=1): drop edges INTO X, clamp X."""
    nf = np.zeros((K, 3), np.float32)
    for i in range(K):
        nf[i] = [bias[i], 1.0 if i == X else 0.0, 1.0 if i == Y else 0.0]
    edges = []                                              # (src j, dst i, weight)
    for i in range(K):
        if i == X:                                         # intervention: cut incoming edges to X
            continue
        for j in range(i):
            if parents[i][j]: edges.append((j, i, w[i][j]))
    return nf, edges, X, Y


class GNNdo(nn.Module):
    def __init__(self, rounds=8):
        super().__init__()
        self.rounds = rounds
        self.embed = nn.Linear(3, Hd)
        self.msg = nn.Sequential(nn.Linear(Hd + 1, Hd), nn.GELU(), nn.Linear(Hd, Hd))
        self.upd = nn.GRUCell(Hd, Hd)
        self.read = nn.Sequential(nn.Linear(Hd, Hd), nn.GELU(), nn.Linear(Hd, 1))
        self.xclamp = nn.Parameter(torch.randn(Hd) * 0.1)

    def forward(self, nf, edges, X, Y):
        K = nf.shape[0]
        h = self.embed(nf)
        h = h.clone(); h[X] = self.xclamp                  # clamp the intervened node
        if edges:
            src = torch.tensor([e[0] for e in edges], device=nf.device)
            dst = torch.tensor([e[1] for e in edges], device=nf.device)
            ew = torch.tensor([[e[2]] for e in edges], dtype=torch.float32, device=nf.device)
        for _ in range(self.rounds):
            if edges:
                m = self.msg(torch.cat([h[src], ew], -1))
                agg = torch.zeros_like(h).index_add_(0, dst, m)
            else:
                agg = torch.zeros_like(h)
            hn = self.upd(agg, h)
            hn = hn.clone(); hn[X] = self.xclamp           # keep intervention clamped each round
            h = hn
        return torch.sigmoid(self.read(h[Y])).squeeze(-1)


def make(Ks, n, seed):
    rng = np.random.default_rng(seed); G = []
    for _ in range(n):
        K = int(rng.choice(Ks)); parents, bias, w, X, Y = sample(K, rng)
        d = do_exact(K, X, Y, parents, bias, w)
        nf, edges, Xi, Yi = to_graph(K, X, Y, parents, bias, w)
        G.append((nf, edges, Xi, Yi, d, K, (parents, bias, w)))
    return G


def evaluate(net, G):
    net.eval(); err = []; obs_err = []
    with torch.no_grad():
        for nf, edges, X, Y, d, K, raw in G:
            t = torch.tensor(nf, device=DEVICE)
            pred = float(net(t, edges, X, Y).cpu())
            err.append(abs(pred - d))
            obs_err.append(abs(obs_cond(K, X, Y, *raw) - d))
    return float(np.mean(err)), float(np.mean(obs_err))


def main():
    torch.manual_seed(0)
    print("=== B4 deepest · SIZE-INVARIANT native do(), EXTRAPOLATE to larger graphs ===\n")
    ntr = 4000 if FAST else 16000
    Gtr = make([4, 5], ntr, 0)
    Gin = make([4, 5], 600, 1)                              # in-distribution test
    G6 = make([6], 500, 2); G7 = make([7], 500, 3)          # EXTRAPOLATION (unseen sizes)
    print(f"  train K∈{{4,5}} ({ntr})  ·  test in-dist K∈{{4,5}}  ·  EXTRAPOLATE K=6, K=7  ·  device={DEVICE}")
    net = GNNdo().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    epochs = 6 if FAST else 18
    for ep in range(epochs):
        net.train(); rng = np.random.default_rng(100 + ep); order = rng.permutation(len(Gtr))
        tot = 0.0
        for c, i in enumerate(order):
            nf, edges, X, Y, d, K, _ = Gtr[i]
            t = torch.tensor(nf, device=DEVICE)
            loss = (net(t, edges, X, Y) - torch.tensor(d, dtype=torch.float32, device=DEVICE)) ** 2
            loss.backward(); tot += float(loss)
            if (c + 1) % 32 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
        if (ep + 1) % max(1, epochs // 3) == 0:
            print(f"    epoch {ep+1}/{epochs}  train-mse {tot/len(Gtr):.5f}")

    mae_in, _ = evaluate(net, Gin)
    mae6, obs6 = evaluate(net, G6)
    mae7, obs7 = evaluate(net, G7)
    print(f"\n  in-distribution  K∈{{4,5}}   native-do MAE = {mae_in:.4f}")
    print(f"  EXTRAPOLATE      K=6        native-do MAE = {mae6:.4f}   (confounded baseline {obs6:.4f})")
    print(f"  EXTRAPOLATE      K=7        native-do MAE = {mae7:.4f}   (confounded baseline {obs7:.4f})")

    thr = 0.06 if FAST else 0.04
    g1 = mae_in < thr
    g2 = mae6 < thr * 1.4 and mae7 < thr * 1.8              # extrapolation degrades gracefully, stays tight
    g3 = mae7 < 0.7 * obs7                                  # still beats the confounded baseline at K=7
    allok = g1 and g2 and g3
    print("\nsize-invariant B4 gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] in-distribution native do engine-tight (MAE<{thr})")
    print(f"  [{'PASS' if g2 else 'FAIL'}] EXTRAPOLATES to unseen larger graphs K=6,7 (degrades gracefully)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] still adjusts: beats confounded baseline at the largest unseen size")
    print(f"\n  >>> {'PASS — one message-passing net internalized the do-calculus ALGORITHM: it EXTRAPOLATES do() to graphs LARGER than any seen in training, engine-tight, structurally faithful (cut-edges + clamp).' if allok else 'CHECK'}")
    print("\nHonest: amortized from given graph+CPTs; exact enumeration stays the recomputable oracle.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
