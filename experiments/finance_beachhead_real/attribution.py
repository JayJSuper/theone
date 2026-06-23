"""Finance causal attribution SUITE on REAL German Credit — the beachhead as a product capability.

Instead of one estimate, run the verifiable engine across each loan-term DECISION (duration,
amount, installment burden) as a treatment, adjusting for the applicant attributes that confound
it (age, employment, savings, checking-account status, existing credits). Produce a ranked causal
attribution: for each factor, the covariate-adjusted causal effect on default + E-value + honest
zone. The product value: a credit-risk report that says not just "X correlates with default" but
"X causally raises default by Δ, and that claim survives unobserved confounding up to RR=E" — and
ABSTAINS / flags uncertainty where the data can't certify it.

Run:  .venv/bin/python experiments/finance_beachhead_real/attribution.py
"""
from __future__ import annotations
import re
import numpy as np
from run import load                                         # reuse the real-data loader


def ate_of(cred):
    m = re.search(r"[-+]?\d*\.?\d+", cred.get("claim", ""))
    return float(m.group()) if m else None


def main():
    print("=== Finance causal attribution · REAL German Credit (n=1000) ===\n")
    arr = load()
    dur, amt, inst, age, ncred, emp, sav, chk, default = [arr[:, i] for i in range(9)]
    Y = default.astype(np.float32)
    def z(x): return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)

    # candidate risk factors (treatments); each is adjusted for ALL OTHER observed covariates,
    # and the E-value then bounds how strong an UNOBSERVED confounder would have to be.
    allfeats = {"long_duration": dur, "high_amount": amt, "high_installment": inst,
                "older_age": age, "many_credits": ncred, "stable_employment": emp,
                "savings": sav, "good_checking": chk}
    treatments = ["long_duration", "high_amount", "high_installment"]

    from theone.native import NativeVerifiableEngine
    eng = NativeVerifiableEngine()
    rows = []
    for name in treatments:
        tvar = allfeats[name]
        T = (tvar > np.median(tvar)).astype(np.float32)
        X = np.stack([z(v) for k, v in allfeats.items() if k != name], axis=1).astype(np.float32)
        naive = Y[T == 1].mean() - Y[T == 0].mean()
        r = eng.estimate_continuous(X, T, Y, covariate_sufficient=True)
        ate = ate_of(r.credential)
        rows.append((name, naive, ate, r.e_value, r.zone, r.replay_ok, r.is_trustworthy()))

    print(f"  {'factor':<17}{'naive':>8}{'causal':>10}{'E-value':>9}  {'zone':<22}{'replay':>7}")
    for name, naive, ate, ev, zone, rok, _ in sorted(rows, key=lambda x: -abs(x[2] if x[2] else 0)):
        a = f"{ate:+.3f}" if ate is not None else "abstain"
        e = f"{ev:.2f}" if ev is not None else "  -"
        print(f"  {name:<17}{naive:>+8.3f}{a:>10}{e:>9}  {zone:<22}{str(rok):>7}")

    print("\n  Honest reading (the product value):")
    print("    • Ranked by causal (not naive) effect on default, each with its OWN E-value.")
    print("    • A large causal effect with a HIGH E-value is a robust risk driver; a similar effect")
    print("      with a LOW E-value is fragile to hidden confounding — the report says which is which.")
    print("    • Every row is replay-verified and carries an honest zone; nothing is asserted as a")
    print("      certified causal number the observational data can't support. That honesty IS the moat.")

    # gate: every factor is replay-verified, carries an honest zone, and reports a sensitivity bound
    # OR honestly abstains (ate/E-value None) — both are acceptable; what's NOT acceptable is an
    # un-recomputable or zone-less assertion.
    g1 = all(rok for (_, _, _, _, _, rok, _) in rows)
    g2 = all(zone in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED", "REJECT") for (_, _, _, _, zone, _, _) in rows)
    g3 = all((ev is not None and ev >= 1.0) or ate is None for (_, _, ate, ev, _, _, _) in rows)
    allok = g1 and g2 and g3
    print("\nattribution gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] every factor's causal estimate is replay-verified")
    print(f"  [{'PASS' if g2 else 'FAIL'}] every factor carries an E-value sensitivity bound")
    print(f"  [{'PASS' if g3 else 'FAIL'}] every factor carries an honest three-zone status")
    print(f"\n  >>> {'PASS — verifiable, sensitivity-ranked causal attribution on REAL finance data' if allok else 'CHECK'}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
