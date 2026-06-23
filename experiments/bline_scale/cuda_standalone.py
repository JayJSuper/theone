"""Self-contained CUDA scaling probe for RunPod — mirrors NativeVerifiableEngine.estimate_
continuous (TARNet ATE + split-half reproducibility-stability) with ZERO repo dependency, so
it bootstraps on a bare PyTorch pod. Sweeps very large N to extend the reproducibility-
stability-vs-scale curve on a datacenter GPU. Prints a parseable RESULT line per N.

Env: THEONE_SCALE_NS (comma N list), THEONE_ATE (true ATE), THEONE_DIM (covariate dim).
"""
import os, time, math
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ATE = float(os.environ.get("THEONE_ATE", "3.0"))
DIM = int(os.environ.get("THEONE_DIM", "12"))
NS = [int(x) for x in os.environ.get("THEONE_SCALE_NS", "100000,400000,1000000").split(",")]


def make(n, ate, seed, d):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logit = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    t = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    base = 2.0 + X[:, 0] + 0.6 * X[:, 2] - 0.4 * X[:, 3] + 0.3 * X[:, 4] * X[:, 5]
    y = (base + ate * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, t, y


def block(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.ELU())


def fit_ite(X, t, y, xm, xs, ym, ys, seed, epochs=400, bs=65536):
    torch.manual_seed(seed)
    phi = nn.Sequential(block(X.shape[1], 200), block(200, 200), block(200, 200))
    h0 = nn.Sequential(block(200, 100), nn.Linear(100, 1))
    h1 = nn.Sequential(block(200, 100), nn.Linear(100, 1))
    net = nn.ModuleList([phi, h0, h1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xn = torch.tensor((X - xm) / xs, device=DEV)
    tt = torch.tensor(t, device=DEV)
    yy = torch.tensor((y - ym) / ys, device=DEV)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            r = phi(Xn[idx])
            pred = torch.where(tt[idx] > 0.5, h1(r).squeeze(1), h0(r).squeeze(1))
            ((pred - yy[idx]) ** 2).mean().backward()
            opt.step()

    def ite(Xq):
        with torch.no_grad():
            r = phi(torch.tensor((Xq - xm) / xs, device=DEV))
            return ((h1(r) - h0(r)).squeeze(1).cpu().numpy()) * ys
    return ite


def estimate(X, t, y, seed=1):
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = y.mean(), y.std() + 1e-6
    ate = float(fit_ite(X, t, y, xm, xs, ym, ys, seed)(X).mean())
    h = len(t) // 2
    a1 = float(fit_ite(X[:h], t[:h], y[:h], xm, xs, ym, ys, seed)(X[:h]).mean())
    a2 = float(fit_ite(X[h:], t[h:], y[h:], xm, xs, ym, ys, seed)(X[h:]).mean())
    stab = 1.0 - min(1.0, abs(a1 - a2) / (abs(ate) + 1e-6))
    return ate, stab


def main():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"THEONE_SCALE device={DEV} gpu={name} true_ate={ATE} dim={DIM} sweep={NS}", flush=True)
    for n in NS:
        X, t, y = make(n, ATE, 0, DIM)
        t0 = time.time()
        ate, stab = estimate(X, t, y)
        dt = time.time() - t0
        print(f"RESULT N={n} ate={ate:.4f} err={abs(ate-ATE):.4f} repro_stability={stab:.4f} sec={dt:.1f}",
              flush=True)
    print("THEONE_SCALE_DONE", flush=True)


if __name__ == "__main__":
    main()
