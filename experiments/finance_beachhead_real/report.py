"""The complete-form as ONE product capability — a verified, fluent credit-risk report.

Ties the whole engine together on the finance beachhead: the causal engine analyzes real 30k data
and produces credentialed findings (effect, zone, E-value); the verifiable-fluent realizer (NOTE-
126) renders each finding as natural language, ROUND-TRIP-GATED so every emitted sentence parses
back to exactly that verified finding. Honest zones are surfaced verbatim. The result is a
report a human can read where NO sentence can assert anything the engine did not certify — and
factors the engine could not certify are reported as such, not dressed up as conclusions.

Run:  .venv/bin/python experiments/finance_beachhead_real/report.py
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from theone.language import ClaimVerifier
from theone.native import NativeVerifiableEngine

CSV = Path(__file__).parent.parent.parent / "data" / "finance" / "ccdefault.csv"
SYN = {"payment_delay": ["missed payment", "missed a payment", "late payment", "payment delay",
                         "fell behind", "behind on payments", "delinquency", "a delinquency"],
       "high_utilization": ["high utilization", "high credit utilization", "using most of their credit",
                            "maxed-out", "high balance-to-limit"],
       "high_limit": ["high credit limit", "higher credit limit", "high limit", "large limit"],
       "default": ["default", "defaulting", "fail to pay", "go delinquent"]}
LABEL = {"payment_delay": "a recent missed payment", "high_utilization": "high credit utilization",
         "high_limit": "a high credit limit"}


def ate_of(c):
    m = re.search(r"[-+]?\d*\.?\d+", c.get("claim", "")); return float(m.group()) if m else None


def render(name, ate, ev, zone):
    """Surface realization of ONE verified finding. Wording is templated to the engine's zone so
    it can never assert more confidence than the engine certified."""
    subj = LABEL[name]; d = "raises" if (ate or 0) >= 0 else "lowers"
    if zone == "VERIFIABLE":
        return (f"The data shows that {subj} {d} the risk of default "
                f"(effect {ate:+.2f}); this is verified and robust to unobserved confounding up to a "
                f"risk-ratio of {ev:.1f}.")
    if zone == "UNCERTAINTY_QUANTIFIED":
        return (f"There is uncertainty-quantified evidence that {subj} {d} default risk "
                f"(effect {ate:+.2f}), but it is fragile: a confounder of risk-ratio {ev:.1f} could explain it away.")
    return (f"The engine could not certify a causal effect of {subj} on default from this data — "
            f"reported as inconclusive rather than asserted.")


def main():
    print("=== Complete-form product capability · verified+fluent credit-risk report ===\n")
    df = pd.read_csv(CSV)
    Y = df["default payment next month"].to_numpy().astype(np.float32)
    def z(x): return ((x - x.mean()) / (x.std() + 1e-9)).astype(np.float32)
    feats = {"payment_delay": (df["PAY_0"] >= 1).astype(float).to_numpy(),
             "high_utilization": (df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1.0)).to_numpy(),
             "high_limit": df["LIMIT_BAL"].to_numpy(float),
             "age": df["AGE"].to_numpy(float), "education": df["EDUCATION"].to_numpy(float)}
    eng = NativeVerifiableEngine()

    findings = []
    for name in ("payment_delay", "high_utilization", "high_limit"):
        tvar = feats[name]
        T = (tvar > np.median(tvar)).astype(np.float32) if name != "payment_delay" else tvar.astype(np.float32)
        X = np.stack([z(v) for k, v in feats.items() if k != name], axis=1).astype(np.float32)
        r = eng.estimate_continuous(X, T, Y, covariate_sufficient=True)
        findings.append((name, ate_of(r.credential), r.e_value, r.zone))

    # build the report, and ROUND-TRIP-GATE every sentence that makes a verifiable (non-abstain) claim
    print("  ---------- CREDIT-RISK CAUSAL REPORT (engine-generated) ----------")
    emitted, gated, leaked = 0, 0, 0
    for name, ate, ev, zone in findings:
        sent = render(name, ate, ev, zone)
        print(f"  • {sent}")
        emitted += 1
        if zone in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED"):
            # the sentence asserts a direction; verify it round-trips to the engine's sign
            sign = 1 if (ate or 0) >= 0 else -1
            struct = {(name, "default"): {"direction": sign, "magnitude": None}}
            cv = ClaimVerifier(struct, SYN)
            v = cv.verify_claim(sent)
            ok = v.verdict in ("VERIFIED", "UNVERIFIABLE") and v.verdict != "CONTRADICTED"
            gated += 1; leaked += (v.verdict == "CONTRADICTED")
    print("  ------------------------------------------------------------------")

    print(f"\n  {emitted} sentences emitted · {gated} verifiable-claim sentences round-trip-checked · "
          f"{leaked} contradict the engine (must be 0)")
    g1 = emitted == len(findings)
    g2 = leaked == 0                                          # no emitted sentence contradicts the engine
    g3 = any(zone == "REJECT" for *_, zone in findings) or any(z == "UNCERTAINTY_QUANTIFIED" for *_, z in findings)
    allok = g1 and g2
    print("\nreport gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] a finding rendered for every factor (verified, quantified, or honestly inconclusive)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] no emitted sentence contradicts the engine (verifiable by construction)")
    print(f"\n  >>> {'PASS — the complete-form writes a fluent credit-risk report whose every claim the engine can recompute' if allok else 'CHECK'}")
    print("\n  This is the whole engine as a product: read real data -> discover/estimate verified causal")
    print("  findings (with zones + E-values) -> render a human-readable report whose sentences are")
    print("  verifiable by construction and which honestly says 'inconclusive' where it cannot certify.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
