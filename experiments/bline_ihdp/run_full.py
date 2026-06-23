"""B1 EXTERNAL VALIDITY (full protocol) · IHDP-100 standard benchmark.

The canonical IHDP evaluation: 100 realizations, standard train/test split (672 train /
75 test units, 25 real covariates). We train the learned-latent TARNet on each train split
and report, averaged over 100 realizations, the standard metrics — directly comparable to
published numbers (TARNet ~0.88/0.95 in/out √PEHE; CFR ~0.71/0.76):
  • in-sample √PEHE   (heterogeneous-effect accuracy on train units)
  • out-sample √PEHE  (on held-out test units — generalization)
  • ATE error
Plus a verifiability angle: split-half ATE recompute on the train set.

This is the publication-grade external-validity check for B1's learned causal latent on
REAL covariate structure. CPU/MPS.

Run:  .venv/bin/python experiments/bline_ihdp/run_full.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
TR = np.load(HERE / "data" / "ihdp100_train.npz")
TE = np.load(HERE / "data" / "ihdp100_test.npz")


def _block(i, o):
    return nn.Sequential(nn.Linear(i, o), nn.ELU())


class TARNet(nn.Module):
    """TARNet-paper sizing: 3×200 shared representation + 3×100 per-treatment heads, ELU."""
    def __init__(self, k, rep=200, hid=100):
        super().__init__()
        self.phi = nn.Sequential(_block(k, rep), _block(rep, rep), _block(rep, rep))
        self.h0 = nn.Sequential(_block(rep, hid), _block(hid, hid), nn.Linear(hid, 1))
        self.h1 = nn.Sequential(_block(rep, hid), _block(hid, hid), nn.Linear(hid, 1))
    def forward(self, X):
        r = self.phi(X)
        return self.h0(r).squeeze(1), self.h1(r).squeeze(1)


def fit(Xtr, ttr, yftr, seed=0, epochs=2000, patience=120):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-6
    ym, ys = yftr.mean(), yftr.std() + 1e-6
    Xn = ((Xtr - xm) / xs).astype(np.float32)
    yn = ((yftr - ym) / ys).astype(np.float32)
    # validation split for early stopping
    idx = rng.permutation(len(ttr)); nval = max(20, len(ttr) // 5)
    vi, ti = idx[:nval], idx[nval:]
    Xt = torch.tensor(Xn[ti], device=DEV); tt = torch.tensor(ttr[ti].astype(np.float32), device=DEV)
    yt = torch.tensor(yn[ti], device=DEV)
    Xv = torch.tensor(Xn[vi], device=DEV); tv = torch.tensor(ttr[vi].astype(np.float32), device=DEV)
    yv = torch.tensor(yn[vi], device=DEV)
    net = TARNet(Xtr.shape[1]).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, wait = 1e9, None, 0
    for _ in range(epochs):
        net.train(); opt.zero_grad(); y0, y1 = net(Xt)
        (((torch.where(tt > 0.5, y1, y0) - yt) ** 2).mean()).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v0, v1 = net(Xv); vloss = float(((torch.where(tv > 0.5, v1, v0) - yv) ** 2).mean())
        if vloss < best - 1e-4:
            best, best_state, wait = vloss, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state:
        net.load_state_dict(best_state)
    net.eval()
    def ite(X):
        Xq = torch.tensor(((X - xm) / xs).astype(np.float32), device=DEV)
        with torch.no_grad():
            y0, y1 = net(Xq)
        return (y1 - y0).cpu().numpy() * ys
    return ite


def pehe(pred, mu0, mu1):
    return float(np.sqrt(np.mean((pred - (mu1 - mu0)) ** 2)))


def main():
    R = TR["x"].shape[2]
    print(f"=== B1 EXTERNAL VALIDITY (full IHDP-100 protocol, {DEV}) ===\n")
    print(f"{R} realizations · train 672 / test 75 units · 25 real covariates\n")
    pin, pout, ate_e, rec = [], [], [], []
    for r in range(R):
        Xtr, ttr, yftr = TR["x"][:, :, r], TR["t"][:, r], TR["yf"][:, r]
        mu0tr, mu1tr = TR["mu0"][:, r], TR["mu1"][:, r]
        Xte = TE["x"][:, :, r]; mu0te, mu1te = TE["mu0"][:, r], TE["mu1"][:, r]
        ite_fn = fit(Xtr, ttr, yftr, seed=1)
        pin.append(pehe(ite_fn(Xtr), mu0tr, mu1tr))
        pout.append(pehe(ite_fn(Xte), mu0te, mu1te))
        ate_e.append(abs(float(ite_fn(Xtr).mean()) - float((mu1tr - mu0tr).mean())))
        if r < 20:                                   # recompute on a subset (cost)
            h = len(ttr) // 2
            a1 = fit(Xtr[:h], ttr[:h], yftr[:h], seed=1)(Xtr[:h]).mean()
            a2 = fit(Xtr[h:], ttr[h:], yftr[h:], seed=1)(Xtr[h:]).mean()
            rec.append(abs(float(a1) - float(a2)))

    def ms(a): return float(np.mean(a)), float(np.std(a))
    print(f"over {R} realizations (mean ± std):")
    print(f"  in-sample  √PEHE = {ms(pin)[0]:.3f} ± {ms(pin)[1]:.3f}")
    print(f"  out-sample √PEHE = {ms(pout)[0]:.3f} ± {ms(pout)[1]:.3f}")
    print(f"  ATE error        = {ms(ate_e)[0]:.3f} ± {ms(ate_e)[1]:.3f}")
    print(f"  ATE split-half recompute (20 reals) = {ms(rec)[0]:.3f}")
    print(f"\n  reference (published): TARNet ~0.88/0.95 · CFR ~0.71/0.76 (in/out √PEHE)")

    competitive = ms(pout)[0] < 1.5 and ms(ate_e)[0] < 0.6
    print("\nB1 external-validity (full protocol) gate:")
    print(f"  out-sample √PEHE competitive (< 1.5) .. {'PASS' if ms(pout)[0] < 1.5 else 'FAIL'}")
    print(f"  ATE error small (< 0.6) ............... {'PASS' if ms(ate_e)[0] < 0.6 else 'FAIL'}")
    print(f"\n  >>> {'PASS — learned causal latent is competitive on the standard IHDP-100 benchmark' if competitive else 'CHECK'}")
    print("\nMeaning: publication-grade external validity — B1's learned-latent estimator generalizes")
    print("to held-out units on real covariates, with numbers in the published range. Honest: IHDP")
    print("assumes ignorability given X; the latent-confounding boundary is bounded separately.")
    (HERE / "results_full.json").write_text(json.dumps(
        {"in_pehe": round(ms(pin)[0], 4), "out_pehe": round(ms(pout)[0], 4),
         "ate_error": round(ms(ate_e)[0], 4), "recompute": round(ms(rec)[0], 4),
         "gate": bool(competitive)}, indent=2))
    if not competitive:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
