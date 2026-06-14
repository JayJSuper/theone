"""Bet ② re-audit (Jack Q-C32): how does counterfactual-gradient transfer degrade
with DISTRIBUTION-SHIFT SEVERITY? The frozen T4 used one cross-family jump
(confounding p (0,0.4)->(0.4,0.8), ratio ~11.7). Here we train once on family A
and test on a SEQUENCE of families at increasing distance, to map the boundary:
within-family transfer is excellent; does cross-family degrade gracefully
(partial transfer, monotone in shift) or collapse (no transfer)?

Pure computation, no API. Reuses the frozen cf-gradient model + dataset builder.
Run: python experiments/t4_shift_severity/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from theone.experiment.cf_gradient import build_dataset_ranged, DualHeadMLP

HERE = Path(__file__).parent
SEEDS = range(12)
EPOCHS = 300
A = dict(p_range=(0.0, 0.4), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))
# test families at increasing confounding-shift distance from A (which is p<=0.4)
TESTS = [
    ("within A (held-out)", dict(p_range=(0.0, 0.4), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
    ("shift+ (0.2,0.5)",    dict(p_range=(0.2, 0.5), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
    ("shift++ (0.3,0.6)",   dict(p_range=(0.3, 0.6), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
    ("shift+++ (0.4,0.7)",  dict(p_range=(0.4, 0.7), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
    ("OOD (0.4,0.8) [frozen B]", dict(p_range=(0.4, 0.8), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
    ("far OOD (0.6,0.9)",   dict(p_range=(0.6, 0.9), bxy_range=(0.0, 0.5), noise_range=(0.1, 0.5))),
]


def main():
    Phi_A, yf_A, yc_A = build_dataset_ranged(512, 400000, **A)
    mu, sd = Phi_A.mean(0), Phi_A.std(0) + 1e-9
    XA = (Phi_A - mu) / sd
    # build test sets (distinct seeds)
    tests = []
    for j, (name, fam) in enumerate(TESTS):
        Phi, yf, yc = build_dataset_ranged(256, 500000 + 1000 * j, **fam)
        X = (Phi - mu) / sd
        base = float(np.mean((yf - yc) ** 2))   # pure-association baseline for this family
        tests.append((name, X, yc, base))
    # train SEEDS models on A, evaluate cf-MSE on each test family
    per = {name: [] for name, *_ in tests}
    for s in SEEDS:
        net = DualHeadMLP(seed=2000 + s)
        net.train(XA, yf_A, yc_A, lam=1.0, epochs=EPOCHS)
        for name, X, yc, base in tests:
            per[name].append(float(np.mean((net.predict_cf(X) - yc) ** 2)))
    within = float(np.median(per["within A (held-out)"]))
    print(f"train family A: p∈(0,0.4). within-family cf-MSE (median) = {within:.5f}\n")
    print(f"{'test family':>26}{'cf-MSE':>10}{'ratio/within':>13}{'baseline':>10}{'vs base':>9}")
    summ = {}
    for name, X, yc, base in tests:
        med = float(np.median(per[name])); ratio = med / max(within, 1e-12)
        vs = "beats" if med < base else "WORSE"
        summ[name] = {"cf_mse": round(med, 5), "ratio_within": round(ratio, 2),
                      "baseline": round(base, 4), "beats_baseline": med < base}
        print(f"{name:>26}{med:>10.5f}{ratio:>13.2f}{base:>10.4f}{vs:>9}")
    (HERE / "results.json").write_text(json.dumps({"train": A, "within_mse": within, "summary": summ}, indent=2))
    print("\nReading: ratio/within shows transfer degrading with shift severity; 'beats base'")
    print("shows whether partial counterfactual logic still transfers (cf-MSE < pure-association).")


if __name__ == "__main__":
    main()
