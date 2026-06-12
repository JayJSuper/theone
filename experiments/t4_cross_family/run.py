"""T4 cross-family transfer judgment (criteria FROZEN: registry T4/T4-S12).
Train cf-dual-head MLP on family A (low-mid confounding), test on family B (high
confounding, OOD). Judge transfer with the frozen T4 gate. No criteria tuning.
Run: python experiments/t4_cross_family/run.py
"""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from theone.experiment.cf_gradient import build_dataset_ranged, DualHeadMLP

HERE = Path(__file__).parent
SEEDS = range(20)
EPOCHS = 300
STRONG_FACTOR = 1.5                     # frozen T4: test <= train * 1.5

# frozen families
A = dict(p_range=(0.0, 0.4), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))
B = dict(p_range=(0.4, 0.8), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))

t0 = time.time()
Phi_A, yf_A, yc_A = build_dataset_ranged(512, 400000, **A)
Phi_Ah, yf_Ah, yc_Ah = build_dataset_ranged(256, 410000, **A)   # held-out A
Phi_B, yf_B, yc_B = build_dataset_ranged(256, 800000, **B)      # OOD test family
mu, sd = Phi_A.mean(0), Phi_A.std(0) + 1e-9
XA, XAh, XB = (Phi_A - mu) / sd, (Phi_Ah - mu) / sd, (Phi_B - mu) / sd

# pure-association baseline intervention MSE on B (predict do-ATE = obs slope)
base_B = float(np.mean((yf_B - yc_B) ** 2))

print("=== T4 cross-family transfer (frozen T4 gate) ===")
print(f"train A: p in {A['p_range']}  test B(OOD): p in {B['p_range']}  seeds={len(list(SEEDS))}")
print(f"pure-association baseline intervention MSE on B = {base_B:.5f}\n")
print(f"{'seed':>5}{'A_holdout_mse':>15}{'B_mse':>11}{'ratio_B/A':>11}{'verdict':>9}")

rows = []
for s in SEEDS:
    net = DualHeadMLP(seed=2000 + s)
    net.train(XA, yf_A, yc_A, lam=1.0, epochs=EPOCHS)
    mse_Ah = float(np.mean((net.predict_cf(XAh) - yc_Ah) ** 2))
    mse_B = float(np.mean((net.predict_cf(XB) - yc_B) ** 2))
    strong = mse_B <= STRONG_FACTOR * mse_Ah
    weak = mse_B < base_B
    verdict = "PASS" if strong else ("WEAK" if weak else "FAIL")
    rows.append({"seed": int(s), "mse_A_holdout": mse_Ah, "mse_B": mse_B,
                 "ratio": mse_B / max(mse_Ah, 1e-12), "verdict": verdict})
    print(f"{s:>5}{mse_Ah:>15.5f}{mse_B:>11.5f}{mse_B/max(mse_Ah,1e-12):>11.2f}{verdict:>9}")

mse_B_all = np.array([r["mse_B"] for r in rows])
mse_Ah_all = np.array([r["mse_A_holdout"] for r in rows])
counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in ("PASS", "WEAK", "FAIL")}
median_ratio = float(np.median([r["ratio"] for r in rows]))
# overall verdict by frozen rule on medians
med_B, med_Ah = float(np.median(mse_B_all)), float(np.median(mse_Ah_all))
overall = ("PASS" if med_B <= STRONG_FACTOR * med_Ah
           else "WEAK" if med_B < base_B else "FAIL")

print(f"\nmedian A_holdout MSE={med_Ah:.5f}  median B MSE={med_B:.5f}  "
      f"median ratio={median_ratio:.2f}  baseline_B={base_B:.5f}")
print(f"per-seed verdicts: {counts}")
print(f"OVERALL T4 VERDICT (frozen gate) = {overall}")
print("T4 ironclad rule: cross-family must pass WEAK to scale up; "
      f"{'OK to consider scaling' if overall in ('PASS','WEAK') else 'DO NOT scale - log to Beautiful Failures'}")

elapsed = time.time() - t0
results = {"families": {"train_A": A, "test_B": B}, "baseline_B_mse": base_B,
           "strong_factor": STRONG_FACTOR, "median_A_holdout_mse": med_Ah,
           "median_B_mse": med_B, "median_ratio": median_ratio,
           "per_seed_verdict_counts": counts, "overall_verdict": overall,
           "rows": rows, "soul_clause": "model may learn generator noise not "
           "counterfactual logic; B-collapse would reveal memorization"}
(HERE / "results.json").write_text(json.dumps(results, indent=2))
sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()[:16]
(HERE / "fingerprints.json").write_text(json.dumps(
    {"results_sha": sha, "elapsed_sec": round(elapsed, 1)}, indent=2))
print(f"\nelapsed {elapsed:.1f}s  results_sha={sha}")
