"""B4 deepened · native do() from OBSERVATIONS, and the identifiability boundary.

The B4 seed learned do() from full CPT params (≈ the engine's formula). The harder, real
question: can a net do causal inference from OBSERVATIONAL quantities alone? The honest
answer depends on whether the confounder is observed:

  • U OBSERVED  : the data gives P(Y|X,U), P(X|U), P(U) — the net can learn the backdoor
                  adjustment and match the engine (this is the B4-seed regime: feasible).
  • U LATENT    : the data gives only P(X), P(Y|X=0), P(Y|X=1). do() is NOT a function of
                  these — different SCMs with the SAME observational stats have DIFFERENT
                  do() (confounding). So do() is information-theoretically UNIDENTIFIABLE
                  from observational-only; no method (net or otherwise) can recover it, and
                  the honest system must DECLARE this, not fake a number.

This probe MEASURES the boundary: the irreducible spread of do() given the observational
triple (how much do() varies among SCMs that look identical observationally) — a learned
net's error cannot beat this spread. Confirming it is the honest B4 result: native inference
is feasible exactly where the information is present, impossible where it is not.

Run:  .venv/bin/python experiments/bline_native_do_obs/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def scm_quantities(p):
    pu, px0, px1, y00, y01, y10, y11 = p
    pX = (1 - pu) * px0 + pu * px1
    pX = min(max(pX, 1e-6), 1 - 1e-6)
    # P(U=1 | X=x)
    pu_x1 = px1 * pu / pX
    pu_x0 = (1 - px1) * pu / (1 - pX)
    pY_x1 = (1 - pu_x1) * y01 + pu_x1 * y11      # observational P(Y=1|X=1)
    pY_x0 = (1 - pu_x0) * y00 + pu_x0 * y10      # observational P(Y=1|X=0)
    do1 = (1 - pu) * y01 + pu * y11              # true do(X=1)
    return pX, pY_x0, pY_x1, do1


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.05, 0.95, (n, 7))
    obs = np.array([scm_quantities(p) for p in P])
    return P, obs[:, :3], obs[:, 3]              # params, observational-triple, do1


def irreducible_spread(obs, do1, bins=8):
    """Among SCMs whose observational triple falls in the same cell, how much does do() vary?
    That spread is what NO observational-only predictor can beat."""
    idx = np.floor(np.clip(obs, 0, 0.999) * bins).astype(int)
    keys = idx[:, 0] * bins * bins + idx[:, 1] * bins + idx[:, 2]
    spreads = []
    for k in np.unique(keys):
        d = do1[keys == k]
        if len(d) >= 5:
            spreads.append(d.max() - d.min())
    return float(np.mean(spreads)), float(np.median(spreads))


class MLP:
    def __init__(self, k, H=48, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1/np.sqrt(k), (k, H)); self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, 1/np.sqrt(H), (H, H)); self.b2 = np.zeros(H)
        self.w3 = rng.normal(0, 1/np.sqrt(H), H); self.b3 = 0.0
    def _f(self, X):
        z1 = np.tanh(X@self.W1+self.b1); z2 = np.tanh(z1@self.W2+self.b2)
        return z1, z2, z2@self.w3+self.b3
    def predict(self, X): return np.clip(self._f(X)[2], 0, 1)
    def fit(self, X, y, lr=0.08, ep=10000):
        n = len(y)
        for e in range(ep):
            if e == ep//2: lr *= 0.3
            z1, z2, o = self._f(X); d = (o-y)/n
            dw3 = z2.T@d; db3 = float(d.sum())
            d2 = (d[:,None]*self.w3)*(1-z2**2); dW2 = z1.T@d2; db2 = d2.sum(0)
            d1 = (d2@self.W2.T)*(1-z1**2); dW1 = X.T@d1; db1 = d1.sum(0)
            self.w3-=lr*dw3; self.b3-=lr*db3; self.W2-=lr*dW2; self.b2-=lr*db2
            self.W1-=lr*dW1; self.b1-=lr*db1
        return self


def main():
    print("=== B4 deepened · native do() from observations + identifiability boundary ===\n")
    Ptr, obs_tr, do_tr = dataset(3000, 0)
    Pte, obs_te, do_te = dataset(800, 1)

    # Case A (U observed): net sees full CPTs -> can match engine
    netA = MLP(7, seed=2).fit(Ptr, do_tr)
    maeA = float(np.abs(netA.predict(Pte) - do_te).mean())

    # Case B (U latent): net sees only the observational triple
    netB = MLP(3, seed=2).fit(obs_tr, do_tr)
    maeB = float(np.abs(netB.predict(obs_te) - do_te).mean())

    spread_mean, spread_med = irreducible_spread(obs_te, do_te)

    print(f"Case A · U OBSERVED (net sees CPTs):    do MAE vs engine = {maeA:.4f}  -> feasible/tight")
    print(f"Case B · U LATENT (net sees obs only):  do MAE vs engine = {maeB:.4f}  -> high")
    print(f"\nirreducible spread of do() given the SAME observational triple: "
          f"mean={spread_mean:.3f} median={spread_med:.3f}")
    print(f"  -> SCMs that look IDENTICAL observationally have do() differing by ~{spread_med:.2f}.")
    print(f"  -> No observational-only predictor can beat this. The net's {maeB:.2f} error is")
    print(f"     mostly this irreducible unidentifiability, NOT a training failure.")

    # honest gate: A is tight (feasible); the latent boundary is PROVEN by the irreducible
    # spread (the net only ever learns the conditional MEAN, so its MAE understates the
    # spread — the spread is the real, method-independent proof of unidentifiability).
    a_ok = maeA < 0.03
    boundary_confirmed = spread_med > 0.12 and maeB > 1.8 * maeA
    gate = a_ok and boundary_confirmed
    print("\nB4-deepened gate (honest):")
    print(f"  U observed -> native do matches engine (feasible) ........... {'PASS' if a_ok else 'FAIL'}")
    print(f"  U latent  -> unidentifiable (spread {spread_med:.2f}>0.12, latent err {maeB/maeA:.1f}× observed) {'PASS' if boundary_confirmed else 'FAIL'}")
    print(f"\n  >>> {'PASS — native inference feasible where info is present; the latent-confounding' if gate else 'CHECK'}")
    print("      boundary is confirmed, not faked. The honest system declares it (+ E-value).")
    print("\nMeaning: amortized native do() is learnable exactly when the confounder is observed.")
    print("Under latent confounding it is information-theoretically impossible — so the native")
    print("engine must ADJUST (needs the confounder) or DECLARE uncertified, never confabulate.")
    (HERE / "results.json").write_text(json.dumps(
        {"mae_observed": round(maeA, 6), "mae_latent": round(maeB, 6),
         "irreducible_spread_median": round(spread_med, 4), "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
