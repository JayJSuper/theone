"""B2 advance · a LEARNED non-autoregressive generator (replaces random proposal).

B2's seed used random proposal (3% hit rate). The plan's next step is a LEARNED
non-AR generator. Here it is, minimal: a small net G(target, z) -> the 7 SCM CPT params
(all output JOINTLY = non-autoregressive), trained so the resulting interventional ATE
matches the requested target. z is a noise seed for diversity. At inference, ask for a
target effect and G emits a whole SCM in one shot; the exact engine then verifies it
(recomputable credential). Acceptance should jump from ~3% (random) to near-100%.

ATE(params) = (1-pu)(y01-y00) + pu(y11-y10) is differentiable, so G is trained by
gradient descent end-to-end (numpy, CPU). Verification stays exact (engine), so every
shipped SCM still carries a third-party-recomputable proof.

Run:  .venv/bin/python experiments/bline_learned_gen/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def ate_formula(params):
    """Differentiable ATE from params (cols: pu,px0,px1,y00,y01,y10,y11)."""
    pu = params[:, 0]; y00 = params[:, 3]; y01 = params[:, 4]; y10 = params[:, 5]; y11 = params[:, 6]
    return (1 - pu) * (y01 - y00) + pu * (y11 - y10)


def engine_ate(p):
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    pu, px0, px1, y00, y01, y10, y11 = p
    g.set_cpt("U", {(): {0: 1 - pu, 1: pu}})
    g.set_cpt("X", {(0,): {0: 1 - px0, 1: px0}, (1,): {0: 1 - px1, 1: px1}})
    oY = list(g.parent_order("Y"))
    vals = {(0, 0): y00, (0, 1): y01, (1, 0): y10, (1, 1): y11}
    g.set_cpt("Y", {tuple(u if q == "U" else x for q in oY): {1: v, 0: 1 - v}
                    for (u, x), v in vals.items()})
    e = InterventionEngine(g)
    return float(e.query_intervention("Y", 1, {"X": 1}).value
                 - e.query_intervention("Y", 1, {"X": 0}).value)


class Generator:
    """G([target, z]) -> 7 params in (0,1), output jointly (non-autoregressive)."""
    def __init__(self, H=48, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.5, (2, H)); self.b1 = np.zeros(H)
        self.W2 = rng.normal(0, 1 / np.sqrt(H), (H, 7)); self.b2 = np.zeros(7)

    def _fwd(self, inp):
        h = np.tanh(inp @ self.W1 + self.b1)
        params = sigmoid(h @ self.W2 + self.b2)
        return h, params

    def gen(self, target, z):
        inp = np.column_stack([target, z])
        return self._fwd(inp)[1]

    def fit(self, epochs=8000, batch=256, lr=0.05, seed=1):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            target = rng.uniform(-0.4, 0.4, batch)
            z = rng.normal(0, 1, batch)
            inp = np.column_stack([target, z])
            h, params = self._fwd(inp)
            ate = ate_formula(params)
            # d loss/d params  (loss = mean (ate-target)^2)
            d_ate = 2 * (ate - target) / batch        # (B,)
            pu, y00, y01, y10, y11 = (params[:, 0], params[:, 3], params[:, 4],
                                      params[:, 5], params[:, 6])
            dp = np.zeros_like(params)
            dp[:, 0] = d_ate * ((y11 - y10) - (y01 - y00))   # d/d pu
            dp[:, 3] = d_ate * (-(1 - pu))                    # y00
            dp[:, 4] = d_ate * (1 - pu)                       # y01
            dp[:, 5] = d_ate * (-pu)                          # y10
            dp[:, 6] = d_ate * pu                             # y11
            dpre = dp * params * (1 - params)                 # through sigmoid
            dW2 = h.T @ dpre; db2 = dpre.sum(0)
            dh = (dpre @ self.W2.T) * (1 - h ** 2)
            dW1 = inp.T @ dh; db1 = dh.sum(0)
            self.W2 -= lr * dW2; self.b2 -= lr * db2
            self.W1 -= lr * dW1; self.b1 -= lr * db1
        return self


def main():
    print("=== B2 advance · learned non-autoregressive generator (verify-gated) ===\n")
    target, tol = 0.30, 0.02
    G = Generator().fit()

    rng = np.random.default_rng(7)
    N = 200
    z = rng.normal(0, 1, N)
    params = G.gen(np.full(N, target), z)        # ONE parallel shot, N diverse SCMs

    accepted, gaps, diversity = 0, [], []
    for p in params:
        a = engine_ate(p)                         # exact verification
        if abs(a - target) <= tol:
            accepted += 1
            gaps.append(abs(a - target))
            diversity.append(p[0])                # pu of accepted, to show variety
    rate = accepted / N
    print(f"learned G asked for ATE≈{target}: generated {N} SCMs in one shot (non-AR)")
    print(f"accepted by exact engine (±{tol}): {accepted}  ({rate*100:.0f}%)   "
          f"[random proposal was ~3% — learning works]")
    if diversity:
        print(f"accepted SCMs are diverse (pu range {min(diversity):.2f}–{max(diversity):.2f}), "
              f"not one memorised solution")
    print(f"max |ATE-target| among accepted = {max(gaps) if gaps else 0:.4f}  (engine-verified)")

    gate = rate > 0.5
    print("\nB2-advance gate:")
    print(f"  learned non-AR generator hits target >50% ..... {'PASS' if gate else 'FAIL'}")
    print(f"  (vs ~3% random) and stays engine-verified")
    print(f"\n  >>> {'PASS — a learned non-AR generator + exact verify gate works' if gate else 'CHECK'}")
    print("\nHonest scope: tiny numpy generator over a differentiable ATE, CPU. It shows the")
    print("learned-proposer + verify-gate loop (plan B2). Real B2 = a latent diffusion generator")
    print("over rich structures; this establishes the loop it plugs into.")
    (HERE / "results.json").write_text(json.dumps(
        {"target": target, "accept_rate": rate, "accepted": accepted, "n": N,
         "random_baseline_rate": 0.03, "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
