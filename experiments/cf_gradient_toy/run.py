"""T2-04 cf_gradient_toy full run (criteria frozen in prereg.md).
Run: python experiments/cf_gradient_toy/run.py
Emits results.json + curves.json + seed_table.csv + fingerprints; prints vital
signs per condition. Does NOT tune anything to manufacture a vital sign.
"""
from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
import numpy as np
from theone.experiment.cf_gradient import (build_dataset, DualHeadMLP,
    vital_sign_1, baseline_intervention_mse_ci)

HERE = Path(__file__).parent
LAMBDAS = (0.0, 0.5, 1.0, 2.0)
SEEDS = list(range(20))
EPOCHS = 300
N_TRAIN, N_HOLDOUT = 512, 256
TRAIN_SEED, HOLDOUT_SEED = 300000, 700000

t0 = time.time()
Phi_tr, yf_tr, yc_tr = build_dataset(N_TRAIN, TRAIN_SEED)
Phi_ho, yf_ho, yc_ho = build_dataset(N_HOLDOUT, HOLDOUT_SEED)
mu, sd = Phi_tr.mean(0), Phi_tr.std(0) + 1e-9
Xtr, Xho = (Phi_tr - mu) / sd, (Phi_ho - mu) / sd
base_mse, base_lo = baseline_intervention_mse_ci(yf_ho, yc_ho, seed=12345)

print("=== T2-04 cf_gradient_toy (frozen prereg) ===")
print(f"train={N_TRAIN} holdout={N_HOLDOUT} epochs={EPOCHS} seeds={len(SEEDS)}")
print(f"baseline (pure-assoc) holdout intervention MSE = {base_mse:.5f}  "
      f"BCa95 lower = {base_lo:.5f}")

results = {"baseline_intervention_mse": base_mse, "baseline_bca95_lower": base_lo,
           "conditions": {}}
curves = {}
rows = ["lambda,seed,vital1,final_loss_fac,final_loss_cf,holdout_int_mse"]
for lam in LAMBDAS:
    seed_rows, hf_last, hc_last = [], None, None
    for s in SEEDS:
        net = DualHeadMLP(seed=1000 + s)
        hf, hc = net.train(Xtr, yf_tr, yc_tr, lam=lam, epochs=EPOCHS)
        v1 = vital_sign_1(hf, hc)
        imse = float(np.mean((net.predict_cf(Xho) - yc_ho) ** 2))
        seed_rows.append({"seed": s, "vital1": v1, "final_loss_fac": hf[-1],
                          "final_loss_cf": hc[-1], "holdout_intervention_mse": imse})
        rows.append(f"{lam},{s},{int(v1)},{hf[-1]:.6f},{hc[-1]:.6f},{imse:.6f}")
        if s == 0:
            hf_last, hc_last = hf, hc      # keep one representative curve
    imses = np.array([r["holdout_intervention_mse"] for r in seed_rows])
    med = float(np.median(imses))
    cond = {"vital1_fraction": float(np.mean([r["vital1"] for r in seed_rows])),
            "median_holdout_intervention_mse": med,
            "iqr": [float(np.quantile(imses, .25)), float(np.quantile(imses, .75))],
            "vital2_beats_baseline_lower": bool(med < base_lo)}
    results["conditions"][f"lambda={lam}"] = {**cond, "seeds": seed_rows}
    curves[f"lambda={lam}"] = {"factual": hf_last, "counterfactual": hc_last}
    print(f"\nlambda={lam}:  vital1 fraction={cond['vital1_fraction']:.2f}  "
          f"median holdout int MSE={med:.5f}  IQR={cond['iqr'][0]:.5f}..{cond['iqr'][1]:.5f}  "
          f"vital2(beats baseline lower)={cond['vital2_beats_baseline_lower']}")

elapsed = time.time() - t0
(HERE / "results.json").write_text(json.dumps(results, indent=2))
(HERE / "curves.json").write_text(json.dumps(curves, indent=2))
(HERE / "seed_table.csv").write_text("\n".join(rows))

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
fp = {"results_sha": _sha(HERE / "results.json"),
      "seed_table_sha": _sha(HERE / "seed_table.csv"),
      "elapsed_sec": round(elapsed, 1),
      "config": {"lambdas": list(LAMBDAS), "seeds": len(SEEDS), "epochs": EPOCHS,
                 "n_train": N_TRAIN, "n_holdout": N_HOLDOUT,
                 "train_seed": TRAIN_SEED, "holdout_seed": HOLDOUT_SEED}}
(HERE / "fingerprints.json").write_text(json.dumps(fp, indent=2))
print(f"\nelapsed {elapsed:.1f}s (budget 1800s)  artifacts: results.json curves.json "
      f"seed_table.csv fingerprints.json (results_sha={fp['results_sha']})")
