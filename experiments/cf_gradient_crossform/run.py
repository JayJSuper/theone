"""Bet ② future-work (paper §7): counterfactual-gradient transfer ACROSS
FUNCTION FORMS (linear -> nonlinear).

The frozen line (cf_gradient_toy -> t4_shift_severity) tested WITHIN-function-form
shift: train a dual-head cf model on a linear-Gaussian SCM family, test on
linear families at increasing CONFOUNDING distance. cf-MSE degraded monotonically
(1.0 -> 1.5 -> 3.7 -> 9.9 -> 14 -> 30x) but beat the pure-association baseline at
every shift level. Paper §7 boundary: "Within-function-form shift is tested;
across function forms (linear->nonlinear) is future work." This script does that
future work.

DESIGN (isolating the function-form variable)
- Train family A: LINEAR structural equation Y = bxy*X + buy*U + e, with
  confounding p in (0,0.4), bxy in (0,0.5), noise in (0.1,0.5)  -- IDENTICAL to
  the t4_shift_severity training family A.
- Test families: SAME confounding/effect/noise ranges as A (so the marginal SCM
  parameter distribution is held fixed), but the structural equation FORM of Y is
  changed: linear (control) / quadratic / sigmoid(tanh) / interaction(X*U).
  The ONLY thing that shifts across the cross-form tests is the function form.
- For comparability we ALSO re-run the within-form confounding-shift sequence as a
  reference column (linear form, shifted p) so cross-form ratios sit next to the
  known within-form gradient.

COUNTERFACTUAL TRUTH (label) -- the honest gold do-ATE
- The cf label is the average unit-do effect:
      ate = E_{x0~p(X)}[ E[Y|do(X=x0+1)] - E[Y|do(X=x0)] ]
  estimated by interventional sampling (cut U->X, resample U fresh under do).
  For the LINEAR form this reduces to bxy (verified numerically ~0.30 vs bxy=0.3),
  so the task is the SAME estimand the frozen line used; only the data-generating
  function changes. For nonlinear forms the true do-ATE genuinely diverges from
  anything the linear observational features encode -- that is the test.

The cf MODEL, vital_sign_1, and the pure-association baseline are the FROZEN
library objects (theone.experiment.cf_gradient) -- not re-implemented here.

Pure computation, no API. All randomness seeded. File-checkpointed + resumable:
if results.json already has a completed run with the same config fingerprint, it
is loaded instead of recomputed (pass --force to recompute).

Run: .venv/bin/python experiments/cf_gradient_crossform/run.py [--force]
"""
from __future__ import annotations
import sys
import json
import time
import math
import hashlib
from pathlib import Path
import numpy as np

from theone.experiment.cf_gradient import (
    DualHeadMLP, vital_sign_1, baseline_intervention_mse_ci, FEATURE_DIM,
)

HERE = Path(__file__).parent
RESULTS = HERE / "results.json"
FINGERPRINTS = HERE / "fingerprints.json"

# ---- frozen config (matches t4_shift_severity training family A) -------------
SEEDS = list(range(12))            # cf-model init/training seeds
EPOCHS = 300
LAM = 1.0                          # supervised cf head (frozen choice from toy)
N_TRAIN, N_TEST = 512, 256
N_SAMPLES = 500                    # observational samples per SCM instance
TRAIN_SEED = 400000               # same base as t4_shift_severity family A
TEST_SEED_BASE = 500000
ATE_N = 40000                     # interventional sample size for gold do-ATE
B_EXTRA = 0.7                     # nonlinear-term coefficient (fixed, see honesty note)

# Training family A: LINEAR, p in (0,0.4)
FAM_A = dict(form="linear", p_range=(0.0, 0.4), bxy_range=(0.0, 0.5),
             noise_range=(0.1, 0.5))

# Cross-form test families: SAME ranges as A, only the FORM of Y changes.
CROSSFORM_TESTS = [
    ("within-form linear (held-out)", dict(form="linear",      p_range=(0.0, 0.4))),
    ("cross-form sigmoid (tanh)",     dict(form="sigmoid",     p_range=(0.0, 0.4))),
    ("cross-form interaction (X*U)",  dict(form="interaction", p_range=(0.0, 0.4))),
    ("cross-form quadratic (X^2)",    dict(form="quadratic",   p_range=(0.0, 0.4))),
]
# Reference: within-FORM confounding-shift sequence (linear, shifted p) -- the
# known frozen gradient, recomputed here under this script's estimator so the
# numbers are directly comparable to the cross-form rows.
WITHINFORM_SHIFT = [
    ("within-form shift+ (0.2,0.5)",  dict(form="linear", p_range=(0.2, 0.5))),
    ("within-form shift++ (0.3,0.6)", dict(form="linear", p_range=(0.3, 0.6))),
    ("within-form OOD (0.4,0.8)",     dict(form="linear", p_range=(0.4, 0.8))),
]
DEFAULTS = dict(bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))


# --------------------------- SCM with selectable form -------------------------
def _y_of(form, x, u, bxy, buy, noise, eps, b_extra=B_EXTRA):
    """Structural equation for Y under a given function form.
    Nonlinear terms are CENTERED so the marginal mean of Y stays ~0 (E[x^2]~1,
    E[tanh x]~0, E[x*u]~0), keeping observational feature scales comparable."""
    if form == "linear":
        return bxy * x + buy * u + eps
    if form == "quadratic":
        return bxy * x + b_extra * (x ** 2 - 1.0) + buy * u + eps
    if form == "sigmoid":
        return bxy * x + b_extra * np.tanh(x) + buy * u + eps
    if form == "interaction":
        return bxy * x + b_extra * (x * u) + buy * u + eps
    raise ValueError(f"unknown form {form}")


def _sample_obs(form, bux, buy, bxy, noise, n, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(n)
    ex_var = max(1e-9, 1 - bux ** 2)
    x = bux * u + rng.standard_normal(n) * math.sqrt(ex_var)
    eps = rng.standard_normal(n) * max(noise, 1e-9)
    y = _y_of(form, x, u, bxy, buy, noise, eps)
    return u, x, y


def _gold_do_ate(form, bux, buy, bxy, noise, seed, n=ATE_N, delta=1.0):
    """E_{x0~p(X)}[ E[Y|do(X=x0+delta)] - E[Y|do(X=x0)] ].
    Under do(X=val) the edge U->X is cut, so U is resampled fresh, independent of
    the intervened X. Noise term has mean 0 -> dropped from the expectation."""
    rng = np.random.default_rng(seed + 777)
    u0 = rng.standard_normal(n)
    ex_var = max(1e-9, 1 - bux ** 2)
    x0 = bux * u0 + rng.standard_normal(n) * math.sqrt(ex_var)

    def y_mean_under_do(xval):
        uu = rng.standard_normal(n)            # U independent of intervened X
        y = _y_of(form, xval, uu, bxy, buy, noise, 0.0)
        return float(np.mean(y))

    return y_mean_under_do(x0 + delta) - y_mean_under_do(x0)


def _features(form, bux, buy, bxy, noise, seed):
    """Observational summary stats (same 5 features as the frozen line) + the
    factual label (obs slope) + the gold counterfactual label (do-ATE)."""
    u, x, y = _sample_obs(form, bux, buy, bxy, noise, N_SAMPLES, seed)
    cov_xy = float(np.cov(x, y, bias=True)[0, 1])
    var_x = float(np.var(x))
    cov_xu = float(np.cov(x, u, bias=True)[0, 1])
    cov_uy = float(np.cov(u, y, bias=True)[0, 1])
    var_u = float(np.var(u))
    phi = np.array([cov_xy, var_x, cov_xu, cov_uy, var_u], float)
    y_fac = cov_xy / var_x                                   # observational slope
    y_cf = _gold_do_ate(form, bux, buy, bxy, noise, seed)    # gold do-ATE
    return phi, y_fac, y_cf


def build_family(form, n_instances, base_seed, p_range,
                 bxy_range=DEFAULTS["bxy_range"], noise_range=DEFAULTS["noise_range"]):
    rng = np.random.default_rng(base_seed)
    Phi = np.empty((n_instances, FEATURE_DIM))
    yf = np.empty(n_instances)
    yc = np.empty(n_instances)
    for i in range(n_instances):
        p = float(rng.uniform(*p_range))
        b = math.sqrt(p)
        bxy = float(rng.uniform(*bxy_range))
        noise = float(rng.uniform(*noise_range))
        Phi[i], yf[i], yc[i] = _features(form, b, b, bxy, noise, base_seed + 1 + i)
    return Phi, yf, yc


# --------------------------------- run ---------------------------------------
def _config_fp():
    cfg = {"seeds": len(SEEDS), "epochs": EPOCHS, "lam": LAM,
           "n_train": N_TRAIN, "n_test": N_TEST, "n_samples": N_SAMPLES,
           "ate_n": ATE_N, "b_extra": B_EXTRA, "train_seed": TRAIN_SEED,
           "test_seed_base": TEST_SEED_BASE,
           "crossform": [t for _, t in CROSSFORM_TESTS],
           "withinform_shift": [t for _, t in WITHINFORM_SHIFT]}
    blob = json.dumps(cfg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16], cfg


def main():
    force = "--force" in sys.argv
    fp, cfg = _config_fp()
    if RESULTS.exists() and not force:
        old = json.loads(RESULTS.read_text())
        if old.get("config_fp") == fp and old.get("complete"):
            print(f"[resume] results.json already complete (config_fp={fp}); "
                  f"loading. Pass --force to recompute.")
            _print_table(old)
            return

    t0 = time.time()
    # --- training family A (linear) ---
    Phi_A, yf_A, yc_A = build_family(n_instances=N_TRAIN, base_seed=TRAIN_SEED, **FAM_A)
    mu, sd = Phi_A.mean(0), Phi_A.std(0) + 1e-9
    XA = (Phi_A - mu) / sd

    # --- build all test families ---
    all_tests = CROSSFORM_TESTS + WITHINFORM_SHIFT
    tests = []
    for j, (name, fam) in enumerate(all_tests):
        Phi, yf, yc = build_family(n_instances=N_TEST, base_seed=TEST_SEED_BASE + 1000 * j, **fam)
        X = (Phi - mu) / sd
        # pure-association baseline for THIS family (predict do-ATE = obs slope),
        # with frozen-library BCa lower bound
        base_mse, base_lo = baseline_intervention_mse_ci(yf, yc, seed=20000 + j)
        tests.append({"name": name, "X": X, "yc": yc,
                      "base_mse": base_mse, "base_lo": base_lo,
                      "cf_truth_mean": float(np.mean(yc)),
                      "cf_truth_std": float(np.std(yc))})

    # --- train SEEDS cf-models on A, evaluate cf-MSE on each test family ---
    per = {t["name"]: [] for t in tests}
    v1 = {t["name"]: [] for t in tests}
    v1_train = []
    for s in SEEDS:
        net = DualHeadMLP(seed=2000 + s)
        hf, hc = net.train(XA, yf_A, yc_A, lam=LAM, epochs=EPOCHS)
        v1_train.append(bool(vital_sign_1(hf, hc)))
        for t in tests:
            per[t["name"]].append(float(np.mean((net.predict_cf(t["X"]) - t["yc"]) ** 2)))

    within = float(np.median(per["within-form linear (held-out)"]))

    summary = {}
    for t in tests:
        arr = np.array(per[t["name"]])
        med = float(np.median(arr))
        summary[t["name"]] = {
            "cf_mse_median": med,
            "cf_mse_iqr": [float(np.quantile(arr, .25)), float(np.quantile(arr, .75))],
            "ratio_vs_within_form": med / max(within, 1e-12),
            "baseline_assoc_mse": t["base_mse"],
            "baseline_bca95_lower": t["base_lo"],
            "beats_baseline_point": bool(med < t["base_mse"]),
            "beats_baseline_bca_lower": bool(med < t["base_lo"]),
            "cf_truth_mean": t["cf_truth_mean"],
            "cf_truth_std": t["cf_truth_std"],
        }

    out = {
        "config_fp": fp, "config": cfg, "complete": True,
        "elapsed_sec": round(time.time() - t0, 1),
        "within_form_cf_mse": within,
        "train_vital1_fraction": float(np.mean(v1_train)),
        "crossform_names": [n for n, _ in CROSSFORM_TESTS],
        "withinform_names": [n for n, _ in WITHINFORM_SHIFT],
        "summary": summary,
    }
    RESULTS.write_text(json.dumps(out, indent=2))

    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    FINGERPRINTS.write_text(json.dumps(
        {"results_sha": _sha(RESULTS), "config_fp": fp,
         "elapsed_sec": out["elapsed_sec"]}, indent=2))
    _print_table(out)
    print(f"\nelapsed {out['elapsed_sec']}s  artifacts: results.json fingerprints.json "
          f"(results_sha={_sha(RESULTS)})")


def _print_table(out):
    within = out["within_form_cf_mse"]
    s = out["summary"]
    print("=== cf_gradient_crossform: linear-trained cf model, "
          "tested across FUNCTION FORMS ===")
    print(f"train family A: LINEAR, p in (0,0.4); within-form cf-MSE (median) = {within:.5f}")
    print(f"train vital1 fraction = {out['train_vital1_fraction']:.2f}\n")
    hdr = (f"{'test family':>33}{'cf-MSE':>10}{'ratio/within':>13}"
           f"{'assoc-base':>11}{'beats base?':>12}{'truth std':>10}")
    print(hdr)
    print("-" * len(hdr))
    order = out["crossform_names"] + out["withinform_names"]
    for name in order:
        r = s[name]
        beats = "beats" if r["beats_baseline_bca_lower"] else "NO (>=base)"
        print(f"{name:>33}{r['cf_mse_median']:>10.5f}"
              f"{r['ratio_vs_within_form']:>13.2f}"
              f"{r['baseline_assoc_mse']:>11.4f}{beats:>12}"
              f"{r['cf_truth_std']:>10.3f}")
    print("\nReading: 'ratio/within' = cross-form degradation vs the within-form "
          "held-out floor.\n'beats base' = cf-MSE below the pure-association BCa95 "
          "lower bound (partial\ncausal signal still transfers). Compare cross-form "
          "rows (top) to the known\nwithin-form confounding-shift gradient (bottom).")


if __name__ == "__main__":
    main()
