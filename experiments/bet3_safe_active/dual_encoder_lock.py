"""bet③ second layer — dual-encoder sensitivity lock (Jack Q-C37). The IPRG idea
applied to *safety judgments*: do not trust a single encoder's "this arm is safe"
belief; require two INDEPENDENT encoders (each seeing its own independent observation
of the arm) to agree it is safe before executing. If they disagree, the safety belief
is not independently recomputable → fall back to 观复 (a known-safe arm).

This is the safety analogue of the computation pillar's pgmpy cross-check: a single
estimator can be confidently wrong; an independently-recomputed agreement cannot
(without both being wrong the same way). We measure whether the dual-encoder lock
reduces 'misjudged-safe harmful' executions beyond the single-encoder baseline, and
compare it to the third-layer metacognitive switch.

Pure simulation, no API. Run: .venv/bin/python experiments/bet3_safe_active/dual_encoder_lock.py
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
_b = importlib.util.spec_from_file_location("b3", HERE / "run.py")
B = importlib.util.module_from_spec(_b); _b.loader.exec_module(B)
_m = importlib.util.spec_from_file_location("mc", HERE / "metacog_switch.py")
M = importlib.util.module_from_spec(_m); _m.loader.exec_module(M)

K, T, TAU, SIGMA, PRIOR_VAR = B.K, B.T, B.TAU, B.SIGMA, B.PRIOR_VAR
DELTA = M.DELTA
SEEDS = range(60)


class Encoder:
    def __init__(self):
        self.mean = np.zeros(K); self.var = np.full(K, PRIOR_VAR)
        self.n = np.zeros(K); self.s = np.zeros(K)

    def p_harm(self):
        return B.norm_cdf((TAU - self.mean) / np.maximum(np.sqrt(self.var), 1e-9))

    def update(self, i, y):
        self.n[i] += 1; self.s[i] += y
        prec = 1 / PRIOR_VAR + self.n[i] / SIGMA ** 2
        self.var[i] = 1 / prec; self.mean[i] = (self.s[i] / SIGMA ** 2) / prec


def run(theta, harmful, dual_lock, seed):
    rng = np.random.default_rng(seed)
    A, C = Encoder(), Encoder()          # two independent encoders
    misjudged_harm = 0
    safe_baseline = int(np.argmax(theta))
    for t in range(T):
        pa, pc = A.p_harm(), C.p_harm()
        # believed safe: encoder A alone, or BOTH encoders if the lock is on
        safe_a = pa <= DELTA
        believed = (safe_a & (pc <= DELTA)) if dual_lock else safe_a
        idx = np.where(believed)[0]
        if len(idx) == 0:
            # no arm both-encoders-agree-safe → 观复 (lock) or least-harmful (baseline policy)
            i = safe_baseline if dual_lock else int(np.argmin(pa))
            acted_as_safe = False
        else:
            std_a = np.sqrt(A.var)
            i = int(idx[np.argmax(std_a[idx])])     # most informative agreed-safe arm
            acted_as_safe = True
        # two INDEPENDENT observations of the chosen arm, one per encoder
        ya = rng.normal(theta[i], SIGMA); yc = rng.normal(theta[i], SIGMA)
        if harmful[i] and acted_as_safe:
            misjudged_harm += 1
        A.update(i, ya); C.update(i, yc)
    return misjudged_harm


def main():
    rows = []
    for s in SEEDS:
        rng = np.random.default_rng(1000 + s)
        theta = rng.normal(0, 1, K); harmful = theta <= TAU
        if harmful.sum() in (0, K):
            continue
        single = run(theta, harmful, dual_lock=False, seed=5000 + s)
        dual = run(theta, harmful, dual_lock=True, seed=5000 + s)
        # third-layer metacog (single encoder) for reference, on the same seed
        mh_meta, _, _ = M.run(theta, harmful, metacog=True, seed=5000 + s)
        rows.append({"single": single, "dual_lock": dual, "metacog": mh_meta})
    sa = np.array([r["single"] for r in rows]); dl = np.array([r["dual_lock"] for r in rows])
    me = np.array([r["metacog"] for r in rows])
    print(f"k={K}, T={T}, harmful if effect<={TAU}; {len(rows)} seeds")
    print(f"\n{'policy':>30} | {'misjudged-safe harmful (mean)':>30}")
    print(f"{'safe-active (single encoder)':>30} | {sa.mean():>30.2f}")
    print(f"{'safe-active + metacog switch (L3)':>30} | {me.mean():>30.2f}")
    print(f"{'safe-active + dual-encoder lock (L2)':>30} | {dl.mean():>30.2f}")
    red = 100 * (1 - dl.mean() / max(sa.mean(), 1e-9))
    print(f"\nDual-encoder lock vs single: {sa.mean():.2f} -> {dl.mean():.2f} "
          f"({red:.0f}% fewer confident safety mistakes); fewer-or-equal on "
          f"{int(np.sum(dl <= sa))}/{len(rows)} seeds")
    print("The IPRG idea on safety: requiring two independent encoders to AGREE an arm "
          "is safe before acting caps the confident-safety mistakes a single encoder makes.")
    (HERE / "dual_lock_results.json").write_text(json.dumps(
        {"single": round(float(sa.mean()), 2), "metacog_L3": round(float(me.mean()), 2),
         "dual_lock_L2": round(float(dl.mean()), 2), "reduction_pct": round(red, 1),
         "n_seeds": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
