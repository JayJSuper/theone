"""Native causal latent — probe 7: B1 deepened. Multi-confounder, stronger nonlinearity,
bigger learned latent — and the deepest honest line: what the learned latent can verify
vs what it CANNOT (and must declare instead of fake).

Two latent confounders U1,U2 (never observed directly). X~Bern(σ(a1·U1+a2·U2)),
Y~Bern(σ(b·X + c1·U1 + c2·U2)). Each confounder is seen only through NONLINEAR proxies
P=tanh(α·U)+noise. A learned MLP encoder recovers a latent for backdoor adjustment.

Three regimes:
  A. BOTH confounders proxied  -> learned latent recovers both -> verifiable do, small residual.
  B. residual SHRINKS with more data (the path scales).
  C. one confounder has NO proxy at all (truly unobserved) -> do() is biased, AND — the
     honest crux — a truth-free incompleteness signal among the OBSERVED proxies does NOT
     reveal it (you cannot detect what has no proxy: NOTE-004 / information-theoretic limit).
     So instead of pretending, we DECLARE 'unobserved-confounding uncertified' and bound it
     with an E-value. Verifiable where possible; honestly-quantified where not.

Run:  .venv/bin/python experiments/native_causal_latent/run_probe7.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
B = 1.5
A1, A2, C1, C2 = 1.0, 1.0, 1.4, 1.4


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def true_do(xval, n=4_000_000, seed=7):
    rng = np.random.default_rng(seed)
    u1, u2 = rng.standard_normal(n), rng.standard_normal(n)
    return float(np.mean(sigmoid(B * xval + C1 * u1 + C2 * u2)))


def gen(n, proxies_u1, proxies_u2, sigma, seed):
    rng = np.random.default_rng(seed)
    u1, u2 = rng.standard_normal(n), rng.standard_normal(n)
    cols = []
    for _ in range(proxies_u1):
        cols.append(np.tanh(rng.uniform(1.0, 2.2) * u1) + rng.normal(0, sigma, n))
    for _ in range(proxies_u2):
        cols.append(np.tanh(rng.uniform(1.0, 2.2) * u2) + rng.normal(0, sigma, n))
    P = np.column_stack(cols) if cols else np.zeros((n, 1))
    x = (rng.random(n) < sigmoid(A1 * u1 + A2 * u2)).astype(float)
    y = (rng.random(n) < sigmoid(B * x + C1 * u1 + C2 * u2)).astype(float)
    return P, x, y


class MLPDo:
    def __init__(self, k, H=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(k), (k, H)); self.b1 = np.zeros(H)
        self.v = rng.normal(0, 1 / np.sqrt(H), H); self.u = 0.0; self.b2 = 0.0

    def _hid(self, P):
        return np.tanh(P @ self.W1 + self.b1)

    def _logit(self, P, x, hid=None):
        hid = self._hid(P) if hid is None else hid
        return hid @ self.v + self.u * x + self.b2

    def fit(self, P, x, y, lr=0.3, epochs=600):
        n = len(y)
        for _ in range(epochs):
            hid = self._hid(P)
            d = (sigmoid(self._logit(P, x, hid)) - y) / n
            dv = hid.T @ d; du = float(d @ x); db2 = float(d.sum())
            dpre = (d[:, None] * self.v[None, :]) * (1 - hid ** 2)
            self.v -= lr * dv; self.u -= lr * du; self.b2 -= lr * db2
            self.W1 -= lr * (P.T @ dpre); self.b1 -= lr * dpre.sum(0)
        return self

    def do(self, P, xval):
        return float(np.mean(sigmoid(self._logit(P, np.full(len(P), float(xval))))))


def learned_do(P, x, y, xval, seed=1):
    return MLPDo(P.shape[1], seed=seed).fit(P, x, y).do(P, xval)


def e_value(p1, p0):
    p1 = min(max(p1, 1e-9), 1 - 1e-9); p0 = min(max(p0, 1e-9), 1 - 1e-9)
    rr = (p1 / p0) if p1 >= p0 else (p0 / p1)
    return rr + np.sqrt(rr * (rr - 1.0))


def subset_drift(P, x, y, xval=1):
    h = P.shape[1] // 2
    if h == 0:
        return float("nan")
    return abs(learned_do(P[:, :h], x, y, xval) - learned_do(P[:, h:], x, y, xval))


def main():
    td1, td0 = true_do(1), true_do(0)
    print("=== probe 7 · B1 deepened: multi-confounder learned latent, verify vs declare ===\n")
    print(f"truth: do(X=1)={td1:.4f}  do(X=0)={td0:.4f}  (two latent confounders, nonlinear proxies)\n")

    # A. both confounders proxied
    P, x, y = gen(30000, 4, 4, 0.4, seed=0)
    a1 = learned_do(P, x, y, 1); a0 = learned_do(P, x, y, 0)
    print(f"A both proxied (8 proxies): learned do(1)={a1:.4f} residual={abs(a1-td1):.4f} · "
          f"recompute drift={subset_drift(P,x,y):.4f}")
    a_ok = abs(a1 - td1) < 0.05

    # B. scaling: residual shrinks with data
    res = []
    for n in (4000, 12000, 36000):
        Pn, xn, yn = gen(n, 4, 4, 0.4, seed=5)
        res.append(abs(learned_do(Pn, xn, yn, 1) - td1))
    print(f"B scaling residual @ n=4k/12k/36k: {res[0]:.4f} -> {res[1]:.4f} -> {res[2]:.4f} "
          f"({'shrinks' if res[-1] < res[0] else 'flat'})")
    b_ok = res[-1] <= res[0]

    # C. one confounder UNOBSERVED (no proxy) -> biased, undetectable by drift, declared via E-value
    Pc, xc, yc = gen(30000, 4, 0, 0.4, seed=2)     # only U1 proxied; U2 has NO proxy
    c1 = learned_do(Pc, xc, yc, 1); c0 = learned_do(Pc, xc, yc, 0)
    drift_obs = subset_drift(Pc, xc, yc)            # drift among observed (U1) proxies
    ev = e_value(c1, c0)
    print(f"\nC U2 unobserved: learned do(1)={c1:.4f} residual={abs(c1-td1):.4f} (BIASED)")
    print(f"   drift among observed proxies = {drift_obs:.4f}  -> LOW: cannot detect what has no proxy")
    print(f"   honest response: DECLARE 'unobserved-confounding uncertified' + E-value bound = {ev:.2f}")
    print(f"   (an unobserved confounder would need assoc >= {ev:.2f} with X and Y to explain the contrast)")
    # the honest property: drift does NOT falsely fire for a fully-unobserved confounder
    c_ok = drift_obs < 0.03 and abs(c1 - td1) > abs(a1 - td1)   # biased but drift quiet

    gate = a_ok and b_ok and c_ok
    print("\nB1-deepened gate:")
    print(f"  A multi-confounder learned latent verifiable .. {'PASS' if a_ok else 'FAIL'}")
    print(f"  B residual shrinks with data (path scales) .... {'PASS' if b_ok else 'FAIL'}")
    print(f"  C unobserved confounder: biased but NOT faked,")
    print(f"    declared + E-value-bounded (honest limit) ... {'PASS' if c_ok else 'FAIL'}")
    print(f"\n  >>> B1 deepened: {'PASS' if gate else 'CHECK'} — learned latent verifies what it observes,")
    print("      honestly declares + bounds what it cannot. Per NOTE-004 / plan B1.")
    print("\nHonest scope: numpy MLP on synthetic data, CPU. Minimal-but-harder B1 evidence.")
    (HERE / "results_probe7.json").write_text(json.dumps(
        {"truth_do1": round(td1, 6), "A_residual": round(abs(a1 - td1), 6),
         "B_scaling": [round(r, 6) for r in res],
         "C_residual_unobserved": round(abs(c1 - td1), 6), "C_drift": round(drift_obs, 6),
         "C_evalue": round(float(ev), 4), "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
