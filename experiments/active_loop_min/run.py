"""T2-05 active_loop_min full run (criteria frozen in prereg.md).
Run: python experiments/active_loop_min/run.py
Emits results.json + fingerprints; prints vital judgment. No criteria tuning.
"""
from __future__ import annotations
import json
import time
import hashlib
from math import comb
from pathlib import Path
from theone.experiment.active_loop import run_experiment

HERE = Path(__file__).parent
SEEDS, BUDGET = range(20), 40

t0 = time.time()
r = run_experiment(SEEDS, budget=BUDGET)
elapsed = time.time() - t0

# exact one-sided sign-test p-value for A_better >= observed (n=20, p=0.5)
n, k = r["n_seeds"], r["A_better_than_C_count"]
p_one_sided = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n

print("=== T2-05 active_loop_min (frozen prereg) ===")
print(f"budget={BUDGET} queries  seeds={n}")
print("final |beta_hat - beta_xy| at budget:")
for s in ("A", "B", "C"):
    med = r["final_error_median"][s]; iqr = r["final_error_iqr"][s]
    print(f"  {s}: median={med:.4f}  IQR=[{iqr[0]:.4f}, {iqr[1]:.4f}]")
print(f"A better than C in {k}/{n} seeds  (one-sided sign-test p={p_one_sided:.4f})")
print(f"VITAL (A beats C, frozen >=15/20) = {r['vital_A_beats_C']}")

r["sign_test_p_one_sided"] = p_one_sided
(HERE / "results.json").write_text(json.dumps(r, indent=2))

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
fp = {"results_sha": _sha(HERE / "results.json"), "elapsed_sec": round(elapsed, 2),
      "config": {"seeds": n, "budget": BUDGET}}
(HERE / "fingerprints.json").write_text(json.dumps(fp, indent=2))
print(f"\nelapsed {elapsed:.2f}s  artifacts: results.json fingerprints.json "
      f"(results_sha={fp['results_sha']})")
