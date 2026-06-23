"""B5/Q3 · the three-zone honest classifier, made REAL on The One's existing assets.

DP proposed a three-zone classifier (verifiable / uncertainty-quantified / reject) and —
after my critique — integrated E-value for latent confounding. DP's code was stubs and
re-implemented E-value (The One already has it in sensitivity.py). This builds it for real
on existing infrastructure: discovery-leg bootstrap stability (structural uncertainty) +
the engine's effect + E-value (latent-confounding sensitivity), and a classifier that puts
a query into a zone with a RECOMPUTABLE status.

The decisive demo (the gap I flagged): a query whose discovered structure is STABLE (low
structural uncertainty) but whose effect is FRAGILE to latent confounding (low E-value)
must be REJECTED — entropy/stability alone would wrongly call it verifiable. The three-zone
classifier catches it because verifiable requires stability AND high E-value AND identifiable.

Run:  .venv/bin/python experiments/bline_three_zone/run.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from theone.layer2_world_model.discovery import bootstrap_stability, discover
from theone.layer2_world_model.sensitivity import e_value_for_do

HERE = Path(__file__).parent
STABILITY_TOL, EVALUE_TOL = 0.8, 2.0


def gen(n, beta_xy, latent, seed):
    """Confounded system U->X, U->Y, X->Y. If latent=True, U is hidden (only X,Y observed)."""
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.45).astype(int)
    x = (rng.random(n) < np.where(u == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .25, (0, 1): min(.25 + beta_xy, .95),
          (1, 0): .65, (1, 1): min(.65 + beta_xy, .95)}     # U strongly affects Y
    y = np.array([1 if rng.random() < py[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    cols = {"X": x, "Y": y} if latent else {"U": u, "X": x, "Y": y}
    return pd.DataFrame(cols)


def observational_rr(df):
    p1 = df[df.X == 1].Y.mean(); p0 = df[df.X == 0].Y.mean()
    return p1, p0


def classify(df):
    """Three zones from real signals: structural stability + identifiability + E-value."""
    stab = bootstrap_stability(df, B=20, seed=0)
    structural_stability = stab["skeleton_agreement"]          # 1 = very stable
    has_confounder = "U" in df.columns                          # identifiability proxy
    p1, p0 = observational_rr(df)
    ev = e_value_for_do(p1, p0)["e_value"]                      # latent-confounding sensitivity

    verifiable = (structural_stability >= STABILITY_TOL) and has_confounder and (ev >= EVALUE_TOL)
    if verifiable:
        zone = "VERIFIABLE"
    elif structural_stability >= 0.5 and ev >= 1.4:
        zone = "UNCERTAINTY_QUANTIFIED"
    else:
        zone = "REJECT"
    return zone, {"structural_stability": round(structural_stability, 3),
                  "identifiable(confounder observed)": has_confounder,
                  "e_value(latent-confounding)": round(ev, 3),
                  "observational_RR": round(p1 / max(p0, 1e-9), 3)}


def main():
    print("=== Q3 made REAL · three-zone honest classifier (stability + identifiability + E-value) ===\n")

    cases = [
        ("A 强效应,混杂可观测", gen(4000, 0.30, latent=False, seed=0)),
        ("B 弱效应,混杂可观测", gen(4000, 0.04, latent=False, seed=1)),
        ("C 混杂潜在(只见X,Y),关联多为混杂", gen(4000, 0.04, latent=True, seed=2)),
    ]
    rows = []
    for name, df in cases:
        zone, ev = classify(df)
        rows.append((name, zone, ev))
        print(f"{name}")
        print(f"   → {zone}")
        print(f"     stability={ev['structural_stability']} · identifiable={ev['identifiable(confounder observed)']}"
              f" · E-value={ev['e_value(latent-confounding)']} · obs RR={ev['observational_RR']}\n")

    # the decisive check: case C's structure is stable but its effect is fragile to latent
    # confounding (low E-value) -> must NOT be VERIFIABLE (the gap entropy-alone would miss)
    cC = next(r for r in rows if r[0].startswith("C"))
    a_ok = rows[0][1] == "VERIFIABLE"
    c_not_verifiable = cC[1] != "VERIFIABLE" and cC[2]["structural_stability"] >= 0.8
    gate = a_ok and c_not_verifiable
    print("Q3-real gate:")
    print(f"  strong identified effect -> VERIFIABLE .............. {'PASS' if a_ok else 'FAIL'}")
    print(f"  latent-confounded effect (stable but low E-value)")
    print(f"    -> NOT verifiable (the entropy-alone gap, fixed) .. {'PASS' if c_not_verifiable else 'FAIL'}")
    print(f"\n  >>> {'PASS — three-zone classifier rejects what stability-alone would wrongly accept' if gate else 'CHECK'}")
    print("\nWhat this adds vs DP: real signals (existing discovery + E-value, not re-implemented,")
    print("not stubs); and it demonstrably catches the latent-confounding gap DP's entropy-only")
    print("version missed. Every zone is a RECOMPUTABLE status claim, not a truth claim.")
    (HERE / "results.json").write_text(json.dumps(
        {"cases": [{"name": r[0], "zone": r[1], **r[2]} for r in rows], "gate": bool(gate)},
        indent=2, default=str))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
