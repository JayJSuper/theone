"""Native engine · continuous outcomes on the IHDP real benchmark.

The native verifiable engine extended to CONTINUOUS outcomes (real products: medical /
economic outcomes are rarely binary). Runs on the IHDP benchmark (real covariates) via the
native path: learned TARNet ATE + reproducible-inference replay + continuous three-zone
(continuous E-value). Compares the native ATE to the known true ATE.

Run:  .venv/bin/python experiments/native_continuous/run.py
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from theone.native import NativeVerifiableEngine

DATA = Path(__file__).resolve().parent.parent / "bline_ihdp" / "data"


def load(i):
    d = np.loadtxt(DATA / f"ihdp_{i}.csv", delimiter=",").astype(np.float32)
    return d[:, 5:], d[:, 0], d[:, 1], (d[:, 4] - d[:, 3])    # X, t, yf, true ITE


def main():
    print("=== native engine · continuous outcomes on IHDP (real benchmark) ===\n")
    eng = NativeVerifiableEngine()
    errs, zones, replays = [], [], []
    print(f"{'real':>5} {'native ATE':>11} {'true ATE':>9} {'|err|':>7} {'zone':>22} {'replay':>7}")
    for i in range(1, 6):
        X, t, yf, ite_true = load(i)
        true_ate = float(ite_true.mean())
        r = eng.estimate_continuous(X, t, yf, covariate_sufficient=True, seed=1)
        err = abs(r.effect - true_ate)
        errs.append(err); zones.append(r.zone); replays.append(r.replay_ok)
        print(f"{i:>5} {r.effect:>11.3f} {true_ate:>9.3f} {err:>7.3f} {r.zone:>22} {str(r.replay_ok):>7}")

    mae = float(np.mean(errs))
    print(f"\nmean ATE error over 5 realizations = {mae:.3f}")
    accurate = mae < 0.6
    replay_ok = all(replays)
    print("\nnative-continuous gate:")
    print(f"  native ATE accurate on real continuous data (<0.6) . {'PASS' if accurate else 'FAIL'}")
    print(f"  reproducible-inference replay holds ................ {'PASS' if replay_ok else 'FAIL'}")
    gate = accurate and replay_ok
    print(f"\n  >>> {'PASS — native engine handles continuous outcomes on a REAL benchmark' if gate else 'CHECK'}")
    print("\nMeaning: the native verifiable engine now works on continuous, real-covariate data")
    print("(IHDP), not just binary toy SCMs — a step toward product-grade. Honest: a neural")
    print("estimate's verification is reproducible-inference + three-zone status, not 1e-6")
    print("symbolic recompute; small samples keep reproducibility-stability noisy.")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
