"""Formal MVP-2A driver (handover T2-07): calibration -> freeze delta_min ->
formal frozen phase -> first REAL EG number. Reads the frozen Q-C7 grid; uses the
real backdoor-adjusted method (Q-C7-3). All seeds explicit; artifacts fingerprinted.

Run: python experiments/mvp2a/run_formal_eg.py
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
from theone.bench.runner import run_calibration, run_frozen

HERE = Path(__file__).parent
CFG = json.loads((HERE / "frozen_grid.json").read_text())
G = CFG["grid"]

cal_grid = {**G, "base_seed": CFG["calibration_base_seed"]}
formal_grid = {**G, "base_seed": CFG["formal_base_seed"]}

print("=== MVP-2A formal run (Q-C7 frozen grid) ===")
print(f"grid: {len(G['beta_xy'])} beta_xy x {len(G['products'])} products x "
      f"{len(G['noise'])} noise x {G['instances_per_cell']} = "
      f"{len(G['beta_xy'])*len(G['products'])*len(G['noise'])*G['instances_per_cell']} per set")

# --- calibration (burn-after-use) -> delta_min ---
cal = run_calibration(cal_grid, str(HERE))
print(f"\n[calibration] n={cal['n_instances']}  tau_median={cal['tau_median']:.6f}  "
      f"delta_min_frozen={cal['delta_min_frozen']:.6f}")

# --- formal frozen phase (independent seeds; burned isolation enforced) ---
rep = run_frozen({"grid": formal_grid, "delta_min": cal["delta_min_frozen"],
                  "near_zero_eps": CFG["near_zero_eps"]}, str(HERE))

egd = rep["eg_distribution"]
print(f"\n[formal] instances={rep['grid_total_instances']}")
print("--- FIRST REAL EG NUMBER ---")
if egd:
    print(f"EG median = {egd['median']:.4f}   mean = {egd['mean']:.4f}   "
          f"q10 = {egd['q10']:.4f}   q90 = {egd['q90']:.4f}   (n={egd['n']}, EG>1 => method better)")
print(f"abs err  method  rmse={rep['abs_errors']['method']['rmse']:.6f} "
      f"mae={rep['abs_errors']['method']['mae']:.6f}")
print(f"abs err  baseline rmse={rep['abs_errors']['baseline']['rmse']:.6f} "
      f"mae={rep['abs_errors']['baseline']['mae']:.6f}")
nz = rep["near_zero_regime"]
if nz:
    print(f"near-zero regime (beta_xy=0, Amendment 1, abs-error only): n={nz['n']}  "
          f"method_rmse={nz['method']['rmse']:.6f}  baseline_rmse={nz['baseline']['rmse']:.6f}  "
          f"method_better={nz['method_better']}")
print(f"CONJUNCTIVE VERDICT (EG-A1a) = {rep['verdict']}")
print(f"scope: {rep['scoping_note']}")

# --- provenance bundle with fingerprints ---
def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
bundle = {
    "frozen_grid_sha": _sha(HERE / "frozen_grid.json"),
    "calibration_report": cal,
    "formal_report": rep,
    "burned_list_sha": _sha(HERE / "burned_list.json"),
}
(HERE / "formal_eg_bundle.json").write_text(json.dumps(bundle, indent=2))
print(f"\nartifacts: calibration_report.json, frozen_report.json, burned_list.json, "
      f"formal_eg_bundle.json (frozen_grid_sha={bundle['frozen_grid_sha']})")
