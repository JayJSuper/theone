"""Fusion Phase D · L3 active-inference decision on the spine.

  • VFE engine: gradient descent drives free energy F from ~1.0 to < 0.01, monotone,
    and the descent minimum matches the closed-form optimum μ* (independent recompute).
  • Active-inference loop: a running perceive→infer cycle keeps per-step F low.
  • DecisionLayer ANSWERs a converged minimization (credential certifies convergence,
    regime declares it is NOT 'autonomy'); a non-converged run ABSTAINS.

Run:  .venv/bin/python experiments/fusion_decision/run.py
"""
from __future__ import annotations
import numpy as np

from theone.layer3_decision import VFEEngine, ActiveInferenceLoop, DecisionLayer


def make_problem(d_o=8, d_z=4, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_o, d_z))
    z_true = rng.standard_normal(d_z) * 0.3
    o = W @ z_true                                   # observation in the range of W
    o = o / (np.linalg.norm(o) + 1e-9) * np.sqrt(2.0)  # scale so F0 (at μ=0) ≈ 1.0
    return W, o


def main():
    print("=== Fusion Phase D: L3 active-inference decision (VFE on the spine) ===\n")
    ok = True
    W, o = make_problem()

    eng = VFEEngine(W, beta=1e-3)
    mu, trace = eng.minimize(o, iters=500)
    monotone = bool(np.all(np.diff(trace) <= 1e-9))
    mu_star = eng.closed_form(o)
    gap = float(np.max(np.abs(mu - mu_star)))
    print(f"VFE minimize: F {trace[0]:.4f} -> {trace[-1]:.4f}  | monotone={monotone} | "
          f"gradient-descent μ vs closed-form μ* gap={gap:.2e}")
    vfe_ok = monotone and trace[-1] < 0.01 and gap < 1e-4
    ok &= vfe_ok

    loop = ActiveInferenceLoop(eng, f_threshold=0.01)
    stream = [o + 0.0 for _ in range(5)]            # stationary stream
    fs = loop.run(stream, inner_iters=100)
    print(f"active-inference loop: per-step F = {[round(f, 4) for f in fs]} (stays low)")
    ok &= all(f < 0.01 for f in fs)

    print("\nDecisionLayer on the spine:")
    L3 = DecisionLayer()
    v = L3.run({"W": W, "observation": o})
    if v.is_answer():
        _, info = v.credential.verify()
        print(f"  converged minimization -> ANSWER | F_final={v.credential.value:.4f} | "
              f"regime='{v.credential.regime}' | recompute gap={info.get('gap', 0):.1e}")
    answer_ok = v.is_answer()

    # too few iterations -> not yet converged -> ABSTAIN
    vb = L3.run({"W": W, "observation": o, "iters": 1})
    abstain_ok = not vb.is_answer()
    print(f"  1 iteration (not converged) -> {'ABSTAIN' if abstain_ok else 'ANSWER'}: "
          f"{vb.reason if abstain_ok else 'F=%.3f' % vb.credential.value}")
    ok &= answer_ok and abstain_ok

    print("\nL3 contract (honest scope): the credential certifies the optimization CONVERGED")
    print("(gradient descent reached the closed-form optimum, monotonically) — not 'autonomy'.")
    print("A non-converged run abstains. The free-energy machinery is real; the claim is bounded.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
