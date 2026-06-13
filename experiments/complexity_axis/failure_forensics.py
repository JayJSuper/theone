"""Failure forensics (Q-C21 attack-2 + gatekeeper supplement-1): reverse-engineer
HOW gpt-5.1 fails at high k. SCMs are deterministic from (k,i), so we regenerate
each problem and compute candidate FALLBACK answers, then test whether the wrong
predictions cluster on one of them:

  truth        = P(Y=1|do(X=1)) via full backdoor over ALL k confounders
  observational= P(Y=1|X=1)            (退化: no adjustment at all)
  partial_km1  = backdoor over only U0..U{k-2}  (截断: drop the last confounder)
  naive_indep  = adjust each U separately, average (假设独立的朴素近似)

Turns '45/50 wrong' into 'here is the approximation they silently used'.
$0 — no API, pure recomputation. Run: python .../failure_forensics.py
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("ca", HERE / "run.py")
ca = importlib.util.module_from_spec(spec); sys.argv = ["x"]; spec.loader.exec_module(ca)
from theone.causal.engine import InterventionEngine
TOL = 0.005


def candidates(k, i):
    """Recompute truth + fallback answers for problem (k,i) from its frozen SCM."""
    g = ca.k_graph(k, ca.BASE_SEED + 1000 * k + i)
    eng = InterventionEngine(g)
    truth = eng.query_intervention("Y", 1, {"X": 1}).value
    obs = eng.query_observation("Y", 1, {"X": 1}).value           # no adjustment
    # partial: backdoor over U0..U{k-2} only (drop last confounder) — manual sum
    Us = [f"U{j}" for j in range(k)]
    keep = Us[:-1]
    if keep:
        partial = 0.0
        for combo in itertools.product((1, 0), repeat=len(keep)):
            ev = dict(zip(keep, combo))
            pu = 1.0
            for u, val in ev.items():
                pu *= g.cpt(u)[()][val]
            partial += pu * eng.query_observation("Y", 1, {**ev, "X": 1}).value
    else:
        partial = obs
    return {"truth": round(truth, 6), "obs": round(obs, 6),
            "partial_km1": round(partial, 6)}


def main():
    rows = [json.loads(l) for l in (HERE / "rows.jsonl").read_text().splitlines()
            if l.strip()]
    for K in (5, 6):
        kr = [r for r in rows if r["k"] == K]
        wrong = [r for r in kr if r["gpt51"]["pred"] is not None
                 and abs(r["gpt51"]["pred"] - r["truth"]) > TOL]
        print(f"\n===== k={K} (2^{K}={2**K}) — gpt-5.1 wrong answers: {len(wrong)}/{len(kr)} =====")
        if not wrong:
            continue
        preds = np.array([r["gpt51"]["pred"] for r in wrong])
        truths = np.array([r["truth"] for r in wrong])
        all_truths = np.array([r["truth"] for r in kr])
        # bias direction + correlation with truth
        bias = float(np.mean(preds - truths))
        r_corr = float(np.corrcoef(preds, truths)[0, 1]) if len(preds) > 2 else float("nan")
        print(f"  mean(pred-truth) = {bias:+.4f}  (systematic bias direction)")
        print(f"  corr(pred,truth) = {r_corr:+.3f}  (~0 => random; >0 => residual signal)")
        # DE-CONFOUND: at high k, truths themselves concentrate near 0.5
        # (balanced confounders average out). 'clusters at 0.5' is only a real
        # signature if wrong-pred spread is MUCH tighter than truth spread.
        print(f"  truth std={all_truths.std():.4f} (range [{all_truths.min():.3f},"
              f"{all_truths.max():.3f}]) — high-k truths concentrate near 0.5 ANYWAY")
        # fallback hypothesis matching
        hit = {"obs_noadjust": 0, "partial_km1": 0, "near_constant": 0}
        for r in wrong:
            c = candidates(K, r["i"])
            p = r["gpt51"]["pred"]
            if abs(p - c["obs"]) <= TOL:
                hit["obs_noadjust"] += 1
            if abs(p - c["partial_km1"]) <= TOL:
                hit["partial_km1"] += 1
        # near-constant: do wrong preds cluster?
        hit["pred_std"] = round(float(np.std(preds)), 4)
        hit["pred_median"] = round(float(np.median(preds)), 4)
        print(f"  fallback matches (within tol):")
        print(f"    = observational (no adjustment): {hit['obs_noadjust']}/{len(wrong)}")
        print(f"    = partial (drop 1 confounder):   {hit['partial_km1']}/{len(wrong)}")
        print(f"  wrong-pred spread: std={hit['pred_std']}  median={hit['pred_median']}")
        # interpretation (de-confounded)
        frac_obs = hit["obs_noadjust"] / len(wrong)
        frac_partial = hit["partial_km1"] / len(wrong)
        clusters = hit["pred_std"] < 0.6 * all_truths.std()    # tighter than truths
        if frac_obs >= 0.3:
            print(f"  >>> SIGNATURE: degrades toward OBSERVATIONAL (gives up adjusting)")
        elif frac_partial >= 0.3:
            print(f"  >>> SIGNATURE: silently drops a confounder (partial adjustment)")
        elif clusters:
            print(f"  >>> SIGNATURE: collapses to constant ~{hit['pred_median']} "
                  f"(BEYOND truth concentration — real default)")
        else:
            print(f"  >>> SIGNATURE: SCATTERED degradation — no clean heuristic "
                  f"(obs {frac_obs:.0%}, partial {frac_partial:.0%} ≈ chance); errors "
                  f"std {hit['pred_std']:.3f} vs truth std {all_truths.std():.3f}. "
                  f"No systematic shortcut to calibrate away.")


if __name__ == "__main__":
    main()
