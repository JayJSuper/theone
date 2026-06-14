"""Bet ③ (re-grounded by Jack Q-C36/37/38): the SAFETY PREMIUM of constrained
active causal experimentation — reference is UNCONSTRAINED active, NOT passive.

Classic result (NOT our claim): active info-gain beats random passive on sample
efficiency. Our increment: a safety-constrained active learner uses the current
posterior to PREDICT each candidate intervention's effect and EXCLUDES those
predicted harmful (intervention-planning safety), and treats uncertain-safety
candidates conservatively (metacognitive safety). We measure whether this avoids
executing catastrophic interventions that pure info-maximization would select —
at what cost to learning.

Setup (pure simulation, no API): k candidate interventions do(X_i); true effect
θ_i; 'harmful' if θ_i <= τ (executing it causes harm, counted). A Gaussian
posterior over each θ_i is updated from noisy observations. Policies over budget T:
  - unconstrained-active: pick argmax posterior std (pure info gain)
  - safe-active: exclude candidates with posterior P(θ_i<=τ) > δ (predicted harmful);
    among the rest pick argmax std; if none safe, sample the lowest-harm-prob one
    (观复 fallback). Conservative under uncertainty.
  - passive-random: random pick (sample-efficiency reference)
Metrics: cumulative HARMFUL executions (the safety cost) and estimation error on
the SAFE set (learning quality).

Run: python experiments/bet3_safe_active/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
K = 12          # candidate interventions
T = 60          # execution budget
TAU = -0.5      # harmful if true effect <= TAU
DELTA = 0.30    # predicted-harmful if posterior P(theta<=tau) > delta
SIGMA = 0.7     # observation noise
PRIOR_VAR = 1.0
SEEDS = range(40)


def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(lambda z: __import__("math").erf(z / np.sqrt(2)))(x))


def run_policy(theta, harmful, policy, seed):
    rng = np.random.default_rng(seed)
    # Gaussian posterior per arm: mean 0, var PRIOR_VAR
    post_mean = np.zeros(K); post_var = np.full(K, PRIOR_VAR)
    n = np.zeros(K); sum_obs = np.zeros(K)
    harm_exec = 0
    for t in range(T):
        std = np.sqrt(post_var)
        p_harm = norm_cdf((TAU - post_mean) / np.maximum(std, 1e-9))  # P(theta<=tau)
        if policy == "passive":
            i = int(rng.integers(0, K))
        elif policy == "unconstrained":
            i = int(np.argmax(std))                       # pure info gain
        else:  # safe-active
            safe = np.where(p_harm <= DELTA)[0]
            if len(safe) > 0:
                i = int(safe[np.argmax(std[safe])])       # most informative SAFE arm
            else:
                i = int(np.argmin(p_harm))                # 观复 fallback: safest available
        # execute
        y = rng.normal(theta[i], SIGMA)
        if harmful[i]:
            harm_exec += 1
        n[i] += 1; sum_obs[i] += y
        # Bayesian update (known noise SIGMA, prior N(0,PRIOR_VAR))
        prec = 1 / PRIOR_VAR + n[i] / SIGMA ** 2
        post_var[i] = 1 / prec
        post_mean[i] = (sum_obs[i] / SIGMA ** 2) / prec
    # estimation error on the SAFE arms (the ones we care to estimate well)
    safe_mask = ~harmful
    mse_safe = float(np.mean((post_mean[safe_mask] - theta[safe_mask]) ** 2))
    return harm_exec, mse_safe


def main():
    rows = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        theta = rng.normal(0, 1, K)
        harmful = theta <= TAU
        if harmful.sum() == 0 or harmful.sum() == K:
            continue
        r = {"seed": int(s), "n_harmful_arms": int(harmful.sum())}
        for pol in ("passive", "unconstrained", "safe-active"):
            he, mse = run_policy(theta, harmful, pol, 5000 + s)
            r[pol] = {"harm_exec": he, "mse_safe": round(mse, 4)}
        rows.append(r)
    print(f"k={K} interventions, budget T={T}, harmful if effect<={TAU}; {len(rows)} seeds\n")
    print(f"{'policy':>16} | {'harmful executions':>19} | {'MSE on safe arms':>16}")
    summ = {}
    for pol in ("passive", "unconstrained", "safe-active"):
        he = np.array([r[pol]["harm_exec"] for r in rows])
        mse = np.array([r[pol]["mse_safe"] for r in rows])
        summ[pol] = {"harm_exec_mean": round(float(he.mean()), 2),
                     "harm_exec_median": float(np.median(he)),
                     "mse_safe_mean": round(float(mse.mean()), 4)}
        print(f"{pol:>16} | {he.mean():>10.1f} (med {np.median(he):>4.0f}) | {mse.mean():>16.4f}")
    # safety premium: safe-active vs unconstrained
    he_u = np.array([r["unconstrained"]["harm_exec"] for r in rows])
    he_s = np.array([r["safe-active"]["harm_exec"] for r in rows])
    mse_u = np.array([r["unconstrained"]["mse_safe"] for r in rows])
    mse_s = np.array([r["safe-active"]["mse_safe"] for r in rows])
    print(f"\nSAFETY PREMIUM (safe-active vs unconstrained-active):")
    print(f"  harmful executions: {he_u.mean():.1f} -> {he_s.mean():.1f} "
          f"({100*(1-he_s.mean()/max(he_u.mean(),1e-9)):.0f}% fewer)")
    print(f"  cost in MSE on safe arms: {mse_u.mean():.4f} -> {mse_s.mean():.4f} "
          f"({'+' if mse_s.mean()>mse_u.mean() else ''}{100*(mse_s.mean()/max(mse_u.mean(),1e-9)-1):.0f}%)")
    wins = int(np.sum(he_s < he_u))
    print(f"  safe-active executed fewer harmful on {wins}/{len(rows)} seeds")
    (HERE / "results.json").write_text(json.dumps({"summary": summ, "n_seeds": len(rows),
        "safety_premium": {"harm_u": float(he_u.mean()), "harm_s": float(he_s.mean()),
                           "mse_u": float(mse_u.mean()), "mse_s": float(mse_s.mean()),
                           "wins": wins}}, indent=2))


if __name__ == "__main__":
    main()
