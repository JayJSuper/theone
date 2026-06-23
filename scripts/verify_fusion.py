"""One-command verification of the entire fusion: runs every fusion experiment's
self-test plus the pytest suite, and prints a PASS/FAIL dashboard.

Usage:  .venv/bin/python scripts/verify_fusion.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

EXPERIMENTS = [
    "fusion_spine", "fusion_rehome", "fusion_perception", "fusion_physics",
    "fusion_decision", "fusion_integration", "fusion_continuous_do", "fusion_discovery",
    "fusion_cognitive_update", "fusion_sensitivity", "fusion_memory_legs",
    "fusion_integration_v2", "fusion_llm_adapter", "fusion_cognitive_os", "fusion_pinn",
    "fusion_benchmark", "fusion_temporal_causal", "fusion_calibrator",
    "fusion_aline_service",
]


def run(cmd) -> bool:
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode == 0


def main():
    print("=" * 60)
    print("The One · fusion verification dashboard")
    print("=" * 60)
    results = {}
    for e in EXPERIMENTS:
        path = ROOT / "experiments" / e / "run.py"
        ok = path.exists() and run([PY, str(path)])
        results[e] = ok
        print(f"  experiment {e:<26} {'PASS' if ok else 'FAIL'}")

    print("-" * 60)
    pytest_ok = run([PY, "-m", "pytest", "tests/", "-q"])
    print(f"  pytest tests/                          {'PASS' if pytest_ok else 'FAIL'}")
    print("=" * 60)

    n_pass = sum(results.values())
    all_ok = n_pass == len(EXPERIMENTS) and pytest_ok
    print(f"experiments: {n_pass}/{len(EXPERIMENTS)} pass | pytest: {'pass' if pytest_ok else 'fail'}")
    print(f"OVERALL: {'ALL GREEN' if all_ok else 'FAILURES PRESENT'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
