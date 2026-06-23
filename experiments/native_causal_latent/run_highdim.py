"""Native causal latent — probe 2: the continuous combinatorial cliff, and when
STRUCTURE collapses it.

Probe 1 showed do() stays verifiable for ONE continuous confounder. With k continuous
confounders, naive exact discretization costs m^k (the continuous analogue of the 2^k
cliff). The deep question for a native causal latent: when can structure *collapse*
that load (like variable elimination), and when is it an irreducible wall?

Two regimes, same k continuous confounders U_i ~ N(0,1):
  • SEPARABLE   : Y ~ Bernoulli(σ(b·X + Σ c_i·U_i)).
      Σ c_i·U_i ~ N(0, ‖c‖²), so the true do collapses to a ONE-dimensional integral
      regardless of k. Naive m^k grid is wasteful; structure removes the cliff.
  • NON-SEPARABLE: Y ~ Bernoulli(σ(b·X + Σ c_i·U_i + d·U_0·U_1)).
      The interaction term does NOT collapse — the true do needs a genuine 2-D (or
      higher) integral. Naive m^k is the only exact route over those dims: a real wall.

Lesson for the next-gen model: a native causal latent that just "encodes continuously"
is not enough; it must learn the structure that *collapses* separable load (criterion 1
terminability), AND honestly declare the irreducible interaction load it cannot collapse
(criterion 1's abstain branch). Verified throughout by an independent integrator.

Run:  .venv/bin/python experiments/native_causal_latent/run_highdim.py
"""
from __future__ import annotations
import itertools, json
from pathlib import Path
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.stats import norm

HERE = Path(__file__).parent
B = 1.5


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def cell_reps(m):
    edges = norm.ppf(np.linspace(0, 1, m + 1))
    reps = []
    for i in range(m):
        a_, b_ = edges[i], edges[i + 1]
        reps.append((norm.pdf(a_) - norm.pdf(b_)) / (norm.cdf(b_) - norm.cdf(a_)))
    return np.array(reps)


def naive_do_separable(c, m):
    """do(X=1) = E_U[σ(b + Σ c_i U_i)] by m^k grid (P(cell)=1/m^k)."""
    reps = cell_reps(m); k = len(c)
    acc, n = 0.0, 0
    for combo in itertools.product(range(m), repeat=k):
        u = reps[list(combo)]
        acc += sigmoid(B + float(np.dot(c, u))); n += 1
    return acc / n


def collapsed_do_separable(c):
    """Structure collapses it: Σ c_i U_i ~ N(0, ‖c‖²) → 1-D integral over Z."""
    s = float(np.sqrt(np.sum(np.array(c) ** 2)))
    val, _ = quad(lambda z: sigmoid(B + z) * norm.pdf(z, scale=max(s, 1e-9)), -np.inf, np.inf)
    return val


def naive_do_nonsep(c, d, m):
    """Y depends on Σ c_i U_i + d·U_0·U_1 (interaction). m^k grid."""
    reps = cell_reps(m); k = len(c)
    acc, n = 0.0, 0
    for combo in itertools.product(range(m), repeat=k):
        u = reps[list(combo)]
        z = float(np.dot(c, u)) + d * u[0] * u[1]
        acc += sigmoid(B + z); n += 1
    return acc / n


def truth_do_nonsep(c, d):
    """Irreducible: integrate U0,U1 jointly (2-D), the rest collapse into a Gaussian."""
    c = np.array(c); s_rest = float(np.sqrt(np.sum(c[2:] ** 2))) if len(c) > 2 else 0.0

    def inner(u1, u0):
        base = B + c[0] * u0 + c[1] * u1 + d * u0 * u1
        if s_rest > 1e-9:
            f, _ = quad(lambda z: sigmoid(base + z) * norm.pdf(z, scale=s_rest), -8 * s_rest, 8 * s_rest)
            return f * norm.pdf(u0) * norm.pdf(u1)
        return sigmoid(base) * norm.pdf(u0) * norm.pdf(u1)
    val, _ = dblquad(inner, -6, 6, -6, 6)
    return val


def main():
    print("=== probe 2: continuous combinatorial cliff & structural collapse ===\n")
    print("SEPARABLE  Y=σ(bX+Σc_iU_i): naive m^k grid vs 1-D structural collapse")
    print(f"{'k':>3} {'naive cost m^k':>14} {'naive do':>10} {'collapsed(1-D)':>14} {'|Δ|':>10}")
    rows = []
    m = 6
    for k in (1, 2, 3, 4, 5):
        c = [0.9] * k
        nv = naive_do_separable(c, m)
        col = collapsed_do_separable(c)
        rows.append({"k": k, "cost": m ** k, "naive": round(nv, 6), "collapsed": round(col, 6),
                     "gap": abs(nv - col)})
        print(f"{k:>3} {m**k:>14} {nv:>10.6f} {col:>14.6f} {abs(nv-col):>10.2e}")
    print("  → separable load is an ILLUSORY cliff: structure (Gaussian sum) collapses "
          "m^k to a 1-D integral. The engine of the future should learn this.\n")

    print("NON-SEPARABLE Y=σ(bX+Σc_iU_i+d·U0·U1): the interaction does NOT collapse")
    c2 = [0.9, 0.9]; d = 1.4
    nv2 = naive_do_nonsep(c2, d, m)
    tr2 = truth_do_nonsep(c2, d)
    print(f"  k=2, interaction d={d}: naive(m^2={m**2})={nv2:.6f}  vs  true 2-D integral={tr2:.6f}  "
          f"|Δ|={abs(nv2-tr2):.2e}")
    # show that pretending it's separable (1-D collapse) is WRONG here
    wrong_collapse = collapsed_do_separable(c2)
    print(f"  if we WRONGLY assumed separability (1-D collapse): {wrong_collapse:.6f}  "
          f"→ error {abs(wrong_collapse-tr2):.4f} (the interaction load is irreducible)")
    print("  → genuine interaction = irreducible integration load; only honest options are "
          "exact (m^k / true multi-D quad) or a declared-imprecise estimate. The continuous cliff is real here.\n")

    print("LESSON for the native causal latent (defect-criterion 1):")
    print("  A continuous causal latent must (a) LEARN the structure that collapses separable")
    print("  load — else it wastes m^k — and (b) DETECT & DECLARE irreducible interaction load")
    print("  it cannot collapse, abstaining or flagging imprecision rather than faking it.")
    (HERE / "results_highdim.json").write_text(json.dumps(
        {"separable": rows, "nonseparable": {
            "naive": round(nv2, 6), "truth_2d": round(tr2, 6),
            "wrong_if_assumed_separable": round(wrong_collapse, 6),
            "collapse_error_if_wrong": round(abs(wrong_collapse - tr2), 6)}}, indent=2))


if __name__ == "__main__":
    main()
