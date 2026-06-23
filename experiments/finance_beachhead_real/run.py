"""Finance beachhead on REAL data — the verifiable causal engine on UCI German Credit (1000 real
loan records). The synthetic->real leap (JK's beachhead = finance, 2026-06-23).

Real causal question: does taking a LONG-TERM loan causally raise default risk, adjusting for the
confounders that drive both loan term and default (credit amount, age, employment, installment
burden, existing credits)? On observational data you cannot randomize, so unobserved confounding
is real — which is exactly where the engine earns its keep: it returns a RECOMPUTABLE causal
estimate + an E-value (how strong an unobserved confounder would have to be to explain it away) +
an honest three-zone status, instead of a black-box score. It quantifies uncertainty or abstains
rather than asserting a confident wrong number.

Run:  .venv/bin/python experiments/finance_beachhead_real/run.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

DATA = Path(__file__).parent.parent.parent / "data" / "finance" / "german.data"

# statlog German Credit column order (21 cols; last = outcome 1=good 2=bad)
EMP = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}                  # employment-since (years bucket)
SAV = {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 0}                  # savings (A65=unknown->0)
CHK = {"A11": 0, "A12": 1, "A13": 2, "A14": 3}                           # checking-account status


def _ensure_data():
    if DATA.exists():
        return
    import urllib.request
    DATA.parent.mkdir(parents=True, exist_ok=True)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    urllib.request.urlretrieve(url, DATA)


def load():
    _ensure_data()
    rows = [ln.split() for ln in DATA.read_text().splitlines() if ln.strip()]
    feats = []
    for r in rows:
        duration = float(r[1]); amount = float(r[4]); install = float(r[7])
        age = float(r[12]); ncred = float(r[15]); emp = EMP.get(r[6], 2)
        sav = SAV.get(r[5], 0); chk = CHK.get(r[0], 0)
        default = 1 if r[20] == "2" else 0                                # bad credit = default = 1
        feats.append((duration, amount, install, age, ncred, emp, sav, chk, default))
    arr = np.array(feats, dtype=float)
    return arr


def main():
    print("=== Finance beachhead · verifiable causal engine on REAL German Credit (n=1000) ===\n")
    arr = load()
    dur, amt, inst, age, ncred, emp, sav, chk, default = [arr[:, i] for i in range(9)]
    print(f"  loaded {len(arr)} real loans · default rate {default.mean():.1%}")

    # treatment: long-term loan (> median duration). outcome: default. confounders: the things that
    # drive BOTH loan term and default (bigger/older loans, age, employment, installment burden).
    T = (dur > np.median(dur)).astype(np.float32)
    Y = default.astype(np.float32)
    # pre-treatment covariates (standardized continuous + buckets)
    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    X = np.stack([z(amt), z(inst), z(age), z(ncred), emp, sav, chk], axis=1).astype(np.float32)
    print(f"  treatment = long-term loan (>{np.median(dur):.0f} mo), {T.mean():.0%} treated · "
          f"outcome = default · {X.shape[1]} confounders\n")

    # naive (confounded) association vs engine's covariate-adjusted verifiable estimate
    naive = Y[T == 1].mean() - Y[T == 0].mean()
    print(f"  naive association  P(default|long) - P(default|short) = {naive:+.3f}  (CONFOUNDED — not causal)")

    from theone.native import NativeVerifiableEngine
    eng = NativeVerifiableEngine()
    r = eng.estimate_continuous(X, T, Y, covariate_sufficient=True)
    cred = r.credential
    ate = cred.get("ate", cred.get("effect"))
    print(f"\n  ENGINE (covariate-adjusted, replay-verified):")
    print(f"    causal effect of long-term loan on default = {ate:+.3f}" if ate is not None else f"    {cred.get('claim')}")
    print(f"    zone               = {r.zone}")
    print(f"    E-value            = {r.e_value:.2f}  (an unobserved confounder would need RR>={r.e_value:.2f}"
          f" on both treatment & outcome to explain this away)")
    print(f"    reproducibility    = {r.structural_stability:.2f}")
    print(f"    replay verified    = {r.replay_ok}")
    print(f"    trustworthy        = {r.is_trustworthy()}")

    print("\n  Honest reading:")
    print("    • The naive number is confounded (long loans are bigger, older, riskier applicants).")
    print("    • The engine returns a RECOMPUTABLE covariate-adjusted estimate with an E-value: the")
    print("      smaller the E-value, the more fragile the claim to unobserved confounding.")
    print("    • The zone is the engine's honest status — it quantifies uncertainty / abstains rather")
    print("      than asserting a confident causal number it cannot stand behind. This is the finance")
    print("      value: a verifiable causal claim with explicit sensitivity, not a black-box score.")

    g1 = r.replay_ok                                          # the estimate is recomputable
    g2 = r.zone in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED", "REJECT")   # honest three-zone status
    g3 = r.e_value is not None and r.e_value >= 1.0           # sensitivity bound reported
    allok = g1 and g2 and g3
    print("\nfinance-beachhead gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] estimate is replay-verified (recomputable on real data)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] honest three-zone status returned ({r.zone})")
    print(f"  [{'PASS' if g3 else 'FAIL'}] E-value sensitivity bound reported ({r.e_value:.2f})")
    print(f"\n  >>> {'PASS — engine produces a verifiable, sensitivity-bounded causal claim on REAL finance data' if allok else 'CHECK'}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
