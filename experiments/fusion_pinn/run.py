"""Fusion deepening⑧ · L0 PINN physics-residual constraint.

A flexible polynomial model is fit to noisy SHO observations on [0,4] and asked to
extrapolate to [4,7]. Data-only fitting explodes; adding a physics-residual penalty
(q'' + omega^2 q = 0 at collocation points) pins it to the physical law and it
extrapolates. Honest scope: the benefit is MEASURED and valid only where the system
truly obeys the ODE.

Run:  .venv/bin/python experiments/fusion_pinn/run.py
"""
from __future__ import annotations
from theone.layer0_physics import extrapolation_benefit


def main():
    print("=== Fusion deepening⑧: L0 PINN physics-residual constraint (SHO) ===\n")
    r = extrapolation_benefit(omega=1.0, degree=12, seed=0)
    print(f"extrapolation RMSE on [4,7] (trained on [0,4]):")
    print(f"   data-only (no physics prior) : {r['rmse_data_only']:.4g}")
    print(f"   physics-constrained (PINN)   : {r['rmse_physics']:.4g}")
    print(f"   improvement                  : {100 * r['improvement']:.1f}%")
    print("\nReading: an unconstrained flexible model overfits and extrapolates catastrophically;")
    print("enforcing the physics residual pins it to the law and it extrapolates accurately.")
    print("Honest scope: valid only where the system genuinely obeys the stated ODE — the benefit")
    print("is measured here on the SHO, not assumed for arbitrary cognition (L0 regime discipline).")
    ok = r["improvement"] > 0.5 and r["rmse_physics"] < 0.1
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}  (improvement > 50% and physics RMSE < 0.1)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
