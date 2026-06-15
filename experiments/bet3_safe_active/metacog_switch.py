"""bet③ layer 3 + metacognition pillar (Jack Q-C37 layer 3 / Q-X1 观复): the
metacognitive safe-mode switch — when the learner's own safety judgment is
UNRELIABLE (high posterior uncertainty near the harm threshold), it should ABSTAIN
(switch to passive/观复) rather than act on a confident-but-unreliable belief.

We compare two safety-constrained active learners:
  - safe-active        : executes the most-informative arm it currently BELIEVES safe
                         (P(harm)<δ), regardless of how uncertain that belief is.
  - safe-active+metacog: if the chosen arm's safety belief is UNCERTAIN (posterior
                         std high AND mean near the harm threshold τ), it abstains
                         and observes a known-safe baseline instead (观复).
Key metric: 'misjudged-safe harmful' executions — arms executed BECAUSE believed
safe that were actually harmful (confident safety mistakes). Knowing-what-you-don't-
know should cut these.

Pure simulation, no API. Run: python experiments/bet3_safe_active/metacog_switch.py
"""
from __future__ import annotations
import importlib.util, json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
_b = importlib.util.spec_from_file_location("b3", HERE / "run.py")
B = importlib.util.module_from_spec(_b); _b.loader.exec_module(B)  # K, T, TAU, SIGMA, PRIOR_VAR, norm_cdf

K, T, TAU, SIGMA, PRIOR_VAR = B.K, B.T, B.TAU, B.SIGMA, B.PRIOR_VAR
DELTA = 0.30           # predicted-harmful if P(theta<=tau) > delta
UNCERT_STD = 0.45      # 'uncertain safety' if posterior std above this...
UNCERT_BAND = 0.4      # ...AND posterior mean within this band of the threshold tau
SEEDS = range(60)


def run(theta, harmful, metacog, seed):
    rng = np.random.default_rng(seed)
    post_mean = np.zeros(K); post_var = np.full(K, PRIOR_VAR)
    n = np.zeros(K); sum_obs = np.zeros(K)
    misjudged_harm = 0   # executed believing-safe, but actually harmful
    total_harm = 0
    # a designated known-safe baseline arm (the truly-safest, revealed for 观复 fallback)
    safe_baseline = int(np.argmax(theta))   # highest effect = safest; observing it is harmless
    for t in range(T):
        std = np.sqrt(post_var)
        p_harm = B.norm_cdf((TAU - post_mean) / np.maximum(std, 1e-9))
        believed_safe = np.where(p_harm <= DELTA)[0]
        if len(believed_safe) == 0:
            i = int(np.argmin(p_harm)); acted_as_safe = False
        else:
            i = int(believed_safe[np.argmax(std[believed_safe])])  # most informative believed-safe
            acted_as_safe = True
            if metacog:
                # is this safety belief UNRELIABLE? high std AND mean near threshold
                if std[i] > UNCERT_STD and abs(post_mean[i] - TAU) < UNCERT_BAND:
                    i = safe_baseline          # 观复: abstain, observe a known-safe arm
                    acted_as_safe = False
        y = rng.normal(theta[i], SIGMA)
        if harmful[i]:
            total_harm += 1
            if acted_as_safe:
                misjudged_harm += 1
        n[i] += 1; sum_obs[i] += y
        prec = 1 / PRIOR_VAR + n[i] / SIGMA ** 2
        post_var[i] = 1 / prec; post_mean[i] = (sum_obs[i] / SIGMA ** 2) / prec
    safe_mask = ~harmful
    mse_safe = float(np.mean((post_mean[safe_mask] - theta[safe_mask]) ** 2))
    return misjudged_harm, total_harm, mse_safe


def main():
    rows = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        theta = rng.normal(0, 1, K); harmful = theta <= TAU
        if harmful.sum() in (0, K):
            continue
        r = {}
        for name, mc in (("safe-active", False), ("safe-active+metacog", True)):
            mh, th, mse = run(theta, harmful, mc, 5000 + s)
            r[name] = {"misjudged_harm": mh, "total_harm": th, "mse_safe": round(mse, 4)}
        rows.append(r)
    print(f"k={K}, budget T={T}, harmful if effect<={TAU}; {len(rows)} seeds\n")
    print(f"{'policy':>22} | {'misjudged-safe harmful':>22} | {'total harmful':>13} | {'MSE safe':>9}")
    summ = {}
    for name in ("safe-active", "safe-active+metacog"):
        mh = np.array([r[name]["misjudged_harm"] for r in rows])
        th = np.array([r[name]["total_harm"] for r in rows])
        mse = np.array([r[name]["mse_safe"] for r in rows])
        summ[name] = {"misjudged_mean": round(float(mh.mean()), 2), "total_harm_mean": round(float(th.mean()), 2),
                      "mse_safe_mean": round(float(mse.mean()), 4)}
        print(f"{name:>22} | {mh.mean():>13.2f} (med {np.median(mh):>3.0f}) | {th.mean():>13.2f} | {mse.mean():>9.4f}")
    mh_b = np.array([r["safe-active"]["misjudged_harm"] for r in rows])
    mh_m = np.array([r["safe-active+metacog"]["misjudged_harm"] for r in rows])
    print(f"\nMETACOGNITIVE SAFETY (abstain-under-uncertainty):")
    print(f"  misjudged-safe harmful executions: {mh_b.mean():.2f} -> {mh_m.mean():.2f} "
          f"({100*(1-mh_m.mean()/max(mh_b.mean(),1e-9)):.0f}% fewer confident safety mistakes)")
    print(f"  fewer-or-equal on {int(np.sum(mh_m <= mh_b))}/{len(rows)} seeds")
    (HERE / "metacog_results.json").write_text(json.dumps({"summary": summ, "n_seeds": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
