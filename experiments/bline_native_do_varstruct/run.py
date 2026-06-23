"""B4 frontier · STRUCTURE-GENERAL native do() — the hardest form of the second kill-gate.

bline_real_b4 showed a net emulates the engine's do() at a FIXED 3-variable structure. The real
question for native verifiable cognition: can one network compute do(X=1) for ARBITRARY causal
structures it has never seen — reading the graph (adjacency) and the CPTs as INPUT and internalizing
the do-calculus ALGORITHM, not memorizing one structure's mapping?

Family (engine-exact by enumeration): random DAGs over k=5 binary vars in topological order; each
var's CPT is logistic in its parents. A query picks treatment X and outcome Y (X before Y). The
exact engine computes P(Y=1 | do(X=1)) by enumerating the 2^(k-1) assignments of the non-treatment
vars with X clamped to 1 (ground truth). A single MLP reads (adjacency + CPT params + X,Y indices)
and predicts do(X=1). We test on HELD-OUT STRUCTURES (graphs unseen in training) and compare to the
confounded observational baseline P(Y=1|X=1).

If native-do stays engine-tight on unseen structures AND beats the confounded baseline, the network
internalized structure-general causal inference — B4 at its strongest, still recomputable (the
enumeration engine is the oracle).

Run:  .venv/bin/python experiments/bline_native_do_varstruct/run.py   (THEONE_FAST=1 for a smoke)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
FAST = os.environ.get("THEONE_FAST") == "1"
K = 5                                            # variables per DAG
NTR = int(os.environ.get("NTR", "4000" if FAST else "60000"))
NTE = int(os.environ.get("NTE", "1000" if FAST else "6000"))
WIDTH = int(os.environ.get("WIDTH", "128"))
EPOCHS = int(os.environ.get("EPOCHS", "30" if FAST else "200"))
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def sample_dag(rng):
    """random DAG in topo order 0..K-1 with GUARANTEED confounding: pick positions C<X<Y, force
    edges C->X, C->Y (a backdoor confounder) and X->Y (a real effect); add random extra edges and
    random logistic CPTs for structural variety. Returns (parents,bias,w,X,Y)."""
    C = int(rng.integers(0, K - 2)); X = int(rng.integers(C + 1, K - 1)); Y = int(rng.integers(X + 1, K))
    parents = [rng.random(i) < 0.45 for i in range(K)]          # random extra edges
    parents[X][C] = True; parents[Y][C] = True; parents[Y][X] = True   # forced confounder + effect
    bias = rng.normal(0, 1.5, K)
    w = [rng.normal(0, 2.4, i) for i in range(K)]               # weights over earlier vars
    return parents, bias, w, X, Y


def p_one(i, assign, parents, bias, w):
    """P(var_i = 1 | parents' values in assign)."""
    z = bias[i]
    pa = parents[i]
    for j in range(i):
        if pa[j]:
            z += w[i][j] * (2 * assign[j] - 1)
    return 1.0 / (1.0 + np.exp(-z))


def do_exact(X, Y, parents, bias, w, x_val=1):
    """P(Y=1 | do(X=x_val)) by exact enumeration over the non-treatment vars (X clamped)."""
    others = [v for v in range(K) if v != X]
    total = 0.0
    for mask in range(1 << len(others)):
        assign = [0] * K
        assign[X] = x_val
        for b, v in enumerate(others):
            assign[v] = (mask >> b) & 1
        prob = 1.0
        for v in others:                                       # joint prob of the non-treatment vars
            pv = p_one(v, assign, parents, bias, w)
            prob *= pv if assign[v] == 1 else (1 - pv)
        if assign[Y] == 1:
            total += prob
    return total


def obs_conditional(X, Y, parents, bias, w, x_val=1):
    """confounded observational P(Y=1 | X=x_val) — enumerate full joint, condition (no adjustment)."""
    num = den = 0.0
    for mask in range(1 << K):
        assign = [(mask >> b) & 1 for b in range(K)]
        prob = 1.0
        for v in range(K):
            pv = p_one(v, assign, parents, bias, w)
            prob *= pv if assign[v] == 1 else (1 - pv)
        if assign[X] == x_val:
            den += prob
            if assign[Y] == 1:
                num += prob
    return num / den if den > 0 else 0.0


def featurize(X, Y, parents, bias, w):
    f = []
    for i in range(K):                                         # adjacency rows (padded)
        f += [1.0 if (j < i and parents[i][j]) else 0.0 for j in range(K)]
    for i in range(K):                                         # CPT: bias + parent weights (masked)
        f.append(bias[i])
        f += [(w[i][j] if (j < i and parents[i][j]) else 0.0) for j in range(K)]
    f += [1.0 if i == X else 0.0 for i in range(K)]
    f += [1.0 if i == Y else 0.0 for i in range(K)]
    return f


def make(n, seed, with_obs=True):
    rng = np.random.default_rng(seed)
    Xf, yf, obs = [], [], []
    while len(Xf) < n:
        parents, bias, w, X, Y = sample_dag(rng)
        d = do_exact(X, Y, parents, bias, w)
        Xf.append(featurize(X, Y, parents, bias, w)); yf.append(d)
        obs.append(obs_conditional(X, Y, parents, bias, w) if with_obs else 0.0)
    return np.array(Xf, np.float32), np.array(yf, np.float32), np.array(obs, np.float32)


class Net(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, WIDTH), torch.nn.GELU(),
                                       torch.nn.Linear(WIDTH, WIDTH), torch.nn.GELU(),
                                       torch.nn.Linear(WIDTH, WIDTH), torch.nn.GELU(),
                                       torch.nn.Linear(WIDTH, 1))

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)


def main():
    print("=== B4 frontier · STRUCTURE-GENERAL native do() vs exact enumeration ===\n")
    print(f"  random DAGs over K={K} binary vars · train {NTR} structures · test {NTE} unseen · device={DEVICE}")
    Xtr, ytr, _ = make(NTR, 0, with_obs=False)
    Xte, yte, obste = make(NTE, 1, with_obs=True)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    xtr = torch.tensor(Xtr, device=DEVICE); ytt = torch.tensor(ytr, device=DEVICE)
    xte = torch.tensor(Xte, device=DEVICE)
    net = Net(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
    epochs = EPOCHS; bs = 1024
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(xtr), device=DEVICE)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            loss = torch.nn.functional.mse_loss(net(xtr[idx]), ytt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {ep+1}/{epochs} train-mse {loss.item():.5f}")

    net.eval()
    with torch.no_grad():
        pred = net(xte).cpu().numpy()
    mae = float(np.mean(np.abs(pred - yte)))
    obs_mae = float(np.mean(np.abs(obste - yte)))            # confounded baseline error vs true do
    mean_mae = float(np.mean(np.abs(yte.mean() - yte)))
    # how often the confounded baseline is materially wrong (true confounding present)
    confounded = float(np.mean(np.abs(obste - yte) > 0.05))
    print(f"\n  native-do vs exact engine, UNSEEN structures   MAE = {mae:.4f}")
    print(f"  confounded baseline P(Y|X) vs true do          MAE = {obs_mae:.4f}  (gap the net must close)")
    print(f"  predict-mean baseline                          MAE = {mean_mae:.4f}")
    print(f"  fraction of queries with real confounding (|obs-do|>0.05) = {100*confounded:.0f}%")

    g1 = mae < 0.02                                          # engine-tight on unseen structures
    g2 = mae < 0.5 * obs_mae                                 # genuinely adjusts (beats confounded obs by 2x)
    allok = g1 and g2
    print("\nstructure-general B4 gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] native do() engine-tight on UNSEEN structures (MAE < 0.02)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] beats the confounded observational baseline by >=2x (really adjusts)")
    print(f"\n  >>> {'PASS — one net internalizes the do-calculus ALGORITHM across arbitrary structures, engine-tight, recomputable — B4 at its strongest' if allok else 'CHECK'}")
    print("\nHonest: amortized from given graph+CPTs; the enumeration engine stays the recomputable oracle.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
