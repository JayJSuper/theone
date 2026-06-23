"""B5 improvement · a better truth-free self-abstain signal (no engine oracle needed).

Probe-7/B5 showed subset-drift catches proxy-INCOMPLETENESS but NOT uniform proxy noise —
the native pipeline answered confidently-wrong on noisy data and only the engine audit
caught it. For the native cognition to self-verify WITHOUT an external oracle, it needs a
signal that detects 'my proxies are too noisy to recover the confounder'.

Insight: proxies P_j = tanh(α_j·U) + noise all share U, so they are mutually correlated;
as noise grows, that shared signal drops. The MEAN PAIRWISE CORRELATION among proxies is a
truth-free reliability signal — high = U is recoverable (trust), low = noise-dominated
(abstain). Combined with subset-drift it catches both failure modes the engine-free way.

Gate: the new reliability signal SEPARATES clean from noisy where drift does not, so the
combined self-abstain ABSTAINs on the noisy data the old signal wrongly answered.

Run:  .venv/bin/python experiments/bline_self_abstain/run.py
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


def true_do():
    v, _ = quad(lambda u: sigmoid(B + C * u) * norm.pdf(u), -np.inf, np.inf)
    return float(v)


def gen(n, p, sigma, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    a = rng.uniform(1.0, 2.2, p)
    P = np.tanh(u[:, None] * a[None, :]) + rng.normal(0, sigma, (n, p))
    x = (rng.random(n) < sigmoid(A * u)).astype(float)
    y = (rng.random(n) < sigmoid(B * x + C * u)).astype(float)
    return P, x, y


class _MLP:
    def __init__(self, k, H=16, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(k), (k, H)); self.b1 = np.zeros(H)
        self.v = rng.normal(0, 1 / np.sqrt(H), H); self.u = 0.0; self.b2 = 0.0
    def _h(self, P): return np.tanh(P @ self.W1 + self.b1)
    def _l(self, P, x, h=None):
        h = self._h(P) if h is None else h
        return h @ self.v + self.u * x + self.b2
    def fit(self, P, x, y, lr=0.3, ep=500):
        n = len(y)
        for _ in range(ep):
            h = self._h(P); d = (sigmoid(self._l(P, x, h)) - y) / n
            self.v -= lr*(h.T@d); self.u -= lr*float(d@x); self.b2 -= lr*float(d.sum())
            dpre = (d[:, None]*self.v)*(1-h**2)
            self.W1 -= lr*(P.T@dpre); self.b1 -= lr*dpre.sum(0)
        return self
    def do1(self, P): return float(np.mean(sigmoid(self._l(P, np.ones(len(P))))))


def native_do(P, x, y, seed=1):
    return _MLP(P.shape[1], seed=seed).fit(P, x, y).do1(P)


def subset_drift(P, x, y):
    h = P.shape[1] // 2
    return abs(native_do(P[:, :h], x, y) - native_do(P[:, h:], x, y))


def proxy_reliability(P):
    """Mean off-diagonal |correlation| among proxies — truth-free shared-signal fraction."""
    Pc = P - P.mean(0)
    cc = np.corrcoef(Pc, rowvar=False)
    k = cc.shape[0]
    off = cc[~np.eye(k, dtype=bool)]
    return float(np.mean(np.abs(off)))


def main():
    truth = true_do()
    print("=== B5 improvement · truth-free self-abstain (proxy reliability) ===\n")
    print(f"oracle do(X=1) = {truth:.4f}\n")
    REL_TOL, DRIFT_TOL = 0.45, 0.02

    rows = []
    for name, sigma, seed in [("clean σ=0.4", 0.4, 0), ("noisy σ=1.8", 1.8, 2)]:
        P, x, y = gen(30000, 6, sigma, seed)
        est = native_do(P, x, y)
        drift = subset_drift(P, x, y)
        rel = proxy_reliability(P)
        audit = abs(est - truth)
        old_decision = "ANSWER" if drift <= DRIFT_TOL else "ABSTAIN"
        new_decision = "ANSWER" if (drift <= DRIFT_TOL and rel >= REL_TOL) else "ABSTAIN"
        rows.append((name, est, audit, drift, rel, old_decision, new_decision))
        print(f"{name}: do={est:.4f} audit_vs_engine={audit:.4f}")
        print(f"   drift={drift:.4f} (old signal) · reliability={rel:.3f} (new signal)")
        print(f"   old self-decision={old_decision}  ->  new self-decision={new_decision}\n")

    clean = rows[0]; noisy = rows[1]
    # the win: new signal separates (clean high rel, noisy low rel); combined abstain now
    # ABSTAINs on the noisy data that the old drift-only signal wrongly ANSWERed.
    separates = clean[4] >= REL_TOL > noisy[4]
    catches_bias = (noisy[2] > 0.05 and noisy[6] == "ABSTAIN")
    clean_ok = clean[2] < 0.05 and clean[6] == "ANSWER"
    gate = separates and catches_bias and clean_ok
    print("B5-improvement gate:")
    print(f"  reliability separates clean({clean[4]:.2f}) from noisy({noisy[4]:.2f}) .. {'PASS' if separates else 'FAIL'}")
    print(f"  new self-abstain ABSTAINs on the biased noisy case (engine-free) ...... {'PASS' if catches_bias else 'FAIL'}")
    print(f"  still ANSWERs the clean, accurate case ............................... {'PASS' if clean_ok else 'FAIL'}")
    print(f"\n  >>> {'PASS — native cognition now self-catches the bias the oracle used to catch' if gate else 'CHECK'}")
    print("\nMeaning: the engine audit caught what self-credentials missed (NOTE-074); this adds a")
    print("truth-free reliability signal so the NATIVE pipeline self-abstains without an oracle —")
    print("reducing reliance on the external engine. Honest scope: numpy MLP, CPU, synthetic.")
    (HERE / "results.json").write_text(json.dumps(
        {"truth": round(truth, 6),
         "clean": {"audit": round(clean[2], 4), "drift": round(clean[3], 4), "rel": round(clean[4], 3)},
         "noisy": {"audit": round(noisy[2], 4), "drift": round(noisy[3], 4), "rel": round(noisy[4], 3)},
         "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
