"""任务二 · EG 推出窄域(强基线 + 非线性)。
PREREG 冻结哈希(跑前): dd1844d4cff7547cc40d63855f1a94ffa5096289000c4481c85ddd9a03ffd441
判据见 PREREG.md。method=分层 g-computation;基线 0/1/2 = 未调整OLS / Y~X+U / Y~X+U+X:U。

Run: python experiments/eg_strong_baseline/run.py
"""
from __future__ import annotations
import hashlib, json, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
PREREG_SHA = "dd1844d4cff7547cc40d63855f1a94ffa5096289000c4481c85ddd9a03ffd441"
assert hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() == PREREG_SHA, "PREREG changed!"

EPS = 1e-6
GRID = dict(bx=[0.0, 0.3, 0.6], bu=[0.5, 1.0], bxu=[0.4, 0.8], sigma=[0.1, 0.3])
INST_PER_CELL = 30
N = 2000


def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))


def structural_Y(fam, X, U, bx, bu, bxu, sigma, rng):
    eps = rng.normal(0, sigma, len(X))
    if fam == "L": return bx * X + bu * U + eps
    if fam == "I": return bx * X + bu * U + bxu * (X * U) + eps
    if fam == "N": return sigmoid(3 * (bx * X + bu * U + bxu * X * U - 0.5)) + eps
    raise ValueError(fam)


def true_ate(fam, bx, bu, bxu):
    """E[Y|do X=1]-E[Y|do X=0], marginalize U~Bernoulli(0.5) (X clamped, noise mean 0)."""
    def muY(x):
        vals = []
        for u in (0, 1):
            if fam == "L": y = bx * x + bu * u
            elif fam == "I": y = bx * x + bu * u + bxu * (x * u)
            else: y = sigmoid(3 * (bx * x + bu * u + bxu * x * u - 0.5))
            vals.append(0.5 * y)
        return sum(vals)
    return muY(1) - muY(0)


def sample(fam, bx, bu, bxu, sigma, seed):
    rng = np.random.default_rng(seed)
    U = (rng.random(N) < 0.5).astype(float)
    # confounding: U -> X (P(X=1) rises with U)
    X = (rng.random(N) < (0.3 + 0.4 * U)).astype(float)
    Y = structural_Y(fam, X, U, bx, bu, bxu, sigma, rng)
    return X, U, Y


def est_stratified(X, U, Y):
    """method: g-computation over binary U -> ATE."""
    ate = 0.0
    for u in (0, 1):
        pu = np.mean(U == u)
        m1 = Y[(X == 1) & (U == u)]; m0 = Y[(X == 0) & (U == u)]
        if len(m1) == 0 or len(m0) == 0: return np.nan
        ate += pu * (m1.mean() - m0.mean())
    return ate


def ols_coef(design, Y):
    b, *_ = np.linalg.lstsq(design, Y, rcond=None)
    return b


def est_baseline0(X, U, Y):                       # unadjusted OLS, Y~X
    D = np.column_stack([np.ones_like(X), X]); return ols_coef(D, Y)[1]


def est_baseline1(X, U, Y):                       # covariate-adjusted, Y~X+U -> X coef
    D = np.column_stack([np.ones_like(X), X, U]); return ols_coef(D, Y)[1]


def est_baseline2(X, U, Y):                       # +interaction, Y~X+U+X:U -> do-effect averaged over U
    D = np.column_stack([np.ones_like(X), X, U, X * U]); b = ols_coef(D, Y)
    pu = U.mean()
    return b[1] + b[3] * pu                        # d/dX averaged over U


def main():
    print("=== 任务二 · EG 强基线 + 非线性 ===")
    print(f"PREREG {PREREG_SHA[:16]}…  grid cells={len(GRID['bx'])*len(GRID['bu'])*len(GRID['bxu'])*len(GRID['sigma'])} x {INST_PER_CELL} x 3 families\n")
    fams = ["L", "I", "N"]
    rec = {f: {"method_err": [], **{bn: [] for bn in bnames}} for f in fams}
    eg = {f: {bn: [] for bn in bnames} for f in fams}
    seed = 70000
    for fam in fams:
        for bx, bu, bxu, sigma in itertools.product(GRID["bx"], GRID["bu"], GRID["bxu"], GRID["sigma"]):
            t = true_ate(fam, bx, bu, bxu)
            for _ in range(INST_PER_CELL):
                seed += 1
                X, U, Y = sample(fam, bx, bu, bxu, sigma, seed)
                m = est_stratified(X, U, Y)
                if np.isnan(m): continue
                merr = abs(m - t) + EPS
                for bn, fn in bnames.items():
                    berr = abs(fn(X, U, Y) - t) + EPS
                    eg[fam][bn].append(berr / merr)
                    rec[fam][bn].append(berr)
                rec[fam]["method_err"].append(merr)

    out = {"prereg_sha": PREREG_SHA, "families": {}}
    print(f"{'family':>8}{'baseline':>16}{'EG median':>12}{'EG IQR':>22}{'method_rmse':>13}{'base_rmse':>11}")
    for fam in fams:
        out["families"][fam] = {}
        mr = float(np.sqrt(np.mean(np.square(rec[fam]['method_err']))))
        for bn in bnames:
            e = np.array(eg[fam][bn]); br = float(np.sqrt(np.mean(np.square(rec[fam][bn]))))
            med, q25, q75 = np.median(e), np.percentile(e, 25), np.percentile(e, 75)
            out["families"][fam][bn] = {"eg_median": float(med), "eg_q25": float(q25),
                                        "eg_q75": float(q75), "method_rmse": mr, "base_rmse": br, "n": len(e)}
            print(f"{fam:>8}{bn:>16}{med:>12.2f}   [{q25:>7.2f},{q75:>7.2f}]   {mr:>11.4f}{br:>11.4f}")
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()
    (HERE / "SHA256SUMS").write_text(f"{sha}  results.json\n")
    print(f"\n诚实读数: 看 N 族 × b1_adj/b2_adj_inter 的 EG —— 接近 1 就是 method 在强基线下无独占优势;>1 才是真剩余增益。")
    print(f"results.json sha256={sha[:16]}…")


bnames = {"b0_unadj": est_baseline0, "b1_adj": est_baseline1, "b2_adj_inter": est_baseline2}


if __name__ == "__main__":
    main()
