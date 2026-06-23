"""B4 seed · native cognition core — can a network LEARN to do causal inference that
MATCHES the exact engine? (The second kill-gate, minimal.)

B4's thesis: the cognitive operation (do-calculus) becomes NATIVE — computed inside a
learned network, not by an external engine call — while staying VERIFIABLE, because the
exact engine is the recomputable ground truth the network must match.

Minimal test (amortized causal inference): generate many random confounded SCMs
(U->X, U->Y, X->Y); the exact engine computes do(X=1) for each (ground-truth labels). A
small MLP learns to predict do(X=1) from the SCM's CPT parameters. On HELD-OUT SCMs, does
the network's native do() match the engine to tolerance?

  • If yes: a net can internalize verifiable causal inference (generalizes to unseen
    structures, matches the engine) — B4's path is alive.
  • The honest discipline: the engine remains the recomputable oracle; the native net is
    only trusted where it matches. At inference the net needs no engine call (native), but
    its claim is verifiable against the engine (auditable).

Run:  .venv/bin/python experiments/bline_native_do/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent


def engine_do1(params):
    pu, px0, px1, y00, y01, y10, y11 = params
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 1 - pu, 1: pu}})
    g.set_cpt("X", {(0,): {0: 1 - px0, 1: px0}, (1,): {0: 1 - px1, 1: px1}})
    oY = list(g.parent_order("Y"))
    vals = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}
    g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY): {1: v, 0: 1 - v}
                    for (u, x), v in vals.items()})
    return float(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value)


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.05, 0.95, size=(n, 7))
    y = np.array([engine_do1(p) for p in P])
    return P, y


class MLP:
    """2-hidden-layer regressor (tanh) -> sigmoid output in (0,1)."""
    def __init__(self, k=7, H=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1 / np.sqrt(k), (k, H)); self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (H, H)); self.b2 = np.zeros(H)
        self.w3 = rng.normal(0, 1 / np.sqrt(H), H); self.b3 = 0.0

    def _fwd(self, X):
        z1 = np.tanh(X @ self.W1 + self.b1)
        z2 = np.tanh(z1 @ self.W2 + self.b2)
        out = z2 @ self.w3 + self.b3                     # linear output (target is a
        return z1, z2, out                                # smooth bilinear fn, not sigmoidal)

    def predict(self, X):
        return np.clip(self._fwd(X)[2], 0.0, 1.0)

    def fit(self, X, y, lr=0.08, epochs=12000):
        n = len(y)
        for ep in range(epochs):
            if ep == epochs // 2:
                lr *= 0.3                                  # simple lr decay
            z1, z2, out = self._fwd(X)
            d = (out - y) / n                              # MSE grad, linear output
            dw3 = z2.T @ d; db3 = float(d.sum())
            d2 = (d[:, None] * self.w3[None, :]) * (1 - z2 ** 2)
            dW2 = z1.T @ d2; db2 = d2.sum(0)
            d1 = (d2 @ self.W2.T) * (1 - z1 ** 2)
            dW1 = X.T @ d1; db1 = d1.sum(0)
            self.w3 -= lr * dw3; self.b3 -= lr * db3
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1
        return self


def main():
    print("=== B4 seed · native (in-net) do() vs the exact engine ===\n")
    Xtr, ytr = dataset(3000, seed=0)
    Xte, yte = dataset(800, seed=1)        # held-out SCMs
    print(f"train on {len(ytr)} random SCMs (engine do() = ground truth), test on {len(yte)} held-out\n")

    net = MLP(seed=2).fit(Xtr, ytr)
    pred = net.predict(Xte)
    err = np.abs(pred - yte)
    mae = float(err.mean()); p95 = float(np.percentile(err, 95)); mx = float(err.max())

    # baseline: predict the global mean (what "no learned inference" gets you)
    base_mae = float(np.abs(yte - ytr.mean()).mean())

    print(f"held-out native-do vs engine-do:  MAE={mae:.4f}  p95={p95:.4f}  max={mx:.4f}")
    print(f"baseline (predict mean) MAE={base_mae:.4f}  ->  the net learned real inference, not the mean")
    print(f"sample (engine, native): "
          f"{[(round(float(a),3), round(float(b),3)) for a,b in zip(yte[:5], pred[:5])]}")

    # the discipline: the engine stays the oracle; the net is trusted only where it matches.
    # 'audit' = fraction of held-out where native is within 0.02 of the engine (recheckable).
    audit_ok = float((err <= 0.02).mean())
    print(f"\naudit: {audit_ok*100:.0f}% of native do() within 0.02 of the exact engine")
    print("(at inference the net needs NO engine call — native — but every claim is")
    print(" verifiable against the engine, the recomputable oracle.)")

    learned = mae < 0.5 * base_mae
    accurate = mae < 0.03
    gate = learned and accurate
    print("\nB4-seed gate:")
    print(f"  net learned real inference (beats mean baseline) . {'PASS' if learned else 'FAIL'}")
    print(f"  native do() matches engine (MAE < 0.03) .......... {'PASS' if accurate else 'FAIL'}")
    print(f"\n  >>> B4 seed: {'PASS — a net internalized verifiable causal inference' if gate else 'CHECK'}")
    print("\nHonest scope: amortized do() from full CPT params (the net learns the do-computation),")
    print("numpy CPU. This shows 'native inference matching the engine' is feasible — the hard")
    print("open part (B4 full) is doing it from raw observations + learned structure, not given CPTs.")
    (HERE / "results.json").write_text(json.dumps(
        {"mae": round(mae, 6), "p95": round(p95, 6), "max": round(mx, 6),
         "baseline_mae": round(base_mae, 6), "audit_within_0.02": round(audit_ok, 4),
         "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
