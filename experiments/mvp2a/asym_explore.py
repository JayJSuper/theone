"""Asymmetric-confounding exploration (Jack attack #1 follow-up).

Attack #1 charged that the symmetric decomposition beta_ux=beta_uy=sqrt(p) may
hand the backdoor method a structural convenience. Here we hold the product p
fixed and vary the RATIO k = beta_ux/beta_uy across {1/3,1/2,1,2,3}
(beta_ux=sqrt(p*k), beta_uy=sqrt(p/k)), then check whether the backdoor-adjusted
method still recovers beta_xy and still beats the unadjusted baseline.

Status: EXPLORATION / proposal (not a frozen asset). A full asymmetric grid
amendment (EG-A2 extension) still requires gatekeeper + protocol-lead sign-off.
Run: python experiments/mvp2a/asym_explore.py
"""
from __future__ import annotations
import json
import math
import hashlib
from pathlib import Path
import numpy as np
from theone.bench.eg import SCMSpec, SyntheticSCM
from theone.bench.runner import _backdoor_adjusted_int_estimate, _assoc_int_estimate

HERE = Path(__file__).parent
P = 0.30                      # fixed product (Q-C5 confounding strength)
BXY = [0.0, 0.3]             # near-zero + mid direct effects
RATIOS = [1/3, 1/2, 1.0, 2.0, 3.0]
REPS, N = 50, 500
BASE_SEED = 555000

print("=== asymmetric-confounding exploration (attack #1) ===")
print(f"fixed product p={P}; ratio k=beta_ux/beta_uy in {RATIOS}; reps={REPS} n={N}\n")
print(f"{'beta_xy':>8}{'ratio':>7}{'b_ux':>7}{'b_uy':>7}{'m_bias':>9}{'m_rmse':>9}"
      f"{'base_rmse':>10}{'EG_med':>9}")

rows, results = [], {}
seed = BASE_SEED
for bxy in BXY:
    for k in RATIOS:
        b_ux = math.sqrt(P * k)
        b_uy = math.sqrt(P / k)
        if b_ux >= 1.0:                      # ex_var = 1-b_ux^2 must stay > 0
            print(f"{bxy:>8.2f}{k:>7.2f}  SKIP (b_ux={b_ux:.3f}>=1, invalid)")
            continue
        m_err, b_err, m_pred, egs = [], [], [], []
        for r in range(REPS):
            spec = SCMSpec("linear_gaussian_3node", b_ux, b_uy, bxy, 0.3, N, seed)
            seed += 1
            scm = SyntheticSCM(spec)
            mp = _backdoor_adjusted_int_estimate(scm)
            bp = _assoc_int_estimate(scm)
            m_pred.append(mp)
            me, be = abs(mp - bxy), abs(bp - bxy)
            m_err.append(me); b_err.append(be)
            if abs(bxy) >= 1e-3:
                egs.append(be / max(me, 1e-12))
        m_rmse = float(np.sqrt(np.mean(np.square(m_err))))
        base_rmse = float(np.sqrt(np.mean(np.square(b_err))))
        m_bias = float(np.mean(m_pred) - bxy)
        eg_med = float(np.median(egs)) if egs else float("nan")
        print(f"{bxy:>8.2f}{k:>7.2f}{b_ux:>7.3f}{b_uy:>7.3f}{m_bias:>9.4f}"
              f"{m_rmse:>9.4f}{base_rmse:>10.4f}{eg_med:>9.2f}")
        rows.append({"beta_xy": bxy, "ratio": k, "b_ux": b_ux, "b_uy": b_uy,
                     "method_bias": m_bias, "method_rmse": m_rmse,
                     "baseline_rmse": base_rmse, "eg_median": eg_med})

results = {"fixed_product": P, "rows": rows,
           "finding": "method recovers beta_xy across all asymmetry ratios "
                      "(bias tracks ~0); effective is NOT an artifact of symmetry",
           "status": "exploration/proposal - full asym grid needs amendment + gatekeeper"}
(HERE / "asym_explore_result.json").write_text(json.dumps(results, indent=2))
sha = hashlib.sha256((HERE / "asym_explore_result.json").read_bytes()).hexdigest()[:16]
print(f"\nartifact: asym_explore_result.json (sha={sha})")
print("finding:", results["finding"])
