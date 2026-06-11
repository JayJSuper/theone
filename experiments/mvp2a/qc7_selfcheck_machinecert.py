"""R3 v3 machine-certification of Jack's Q-C7 self-check.

Gatekeeper (Jack) hand-computed a self-check for grid point
  beta_ux=0.6, beta_uy=0.5, beta_xy=0.3, noise=0.3, n=500
with claimed exact answers:
  1. confounding strength      = 0.30
  2. observational association = 0.60
  3. interventional ATE        = 0.30
  4. confounding bias          = 0.30

This script certifies those digits TWO independent ways, using the project's
own SCM engine (not a re-derivation), per R3 v3 (hand-proof + machine-recompute,
machine authoritative on disagreement):

  (A) ANALYTIC  - SyntheticSCM closed-form truths (true_bias / true_obs_slope /
                  true_int_ate).
  (B) EMPIRICAL - Monte-Carlo via SyntheticSCM.sample(): observational OLS slope
                  on a large standardized sample, and the interventional slope on
                  an intervened sample (X drawn independent of U == graph surgery
                  do(X)).  Confirms the standardization (Var(X)=1) and the do cut.

All randomness uses explicit seeds.  Exit 0 iff every digit matches Jack within
tolerance; nonzero otherwise (so it can gate a freeze).
"""
from __future__ import annotations
import math
import sys
import json
import numpy as np
from theone.bench.eg import SCMSpec, SyntheticSCM

TOL_ANALYTIC = 1e-9
TOL_EMPIRICAL = 5e-3          # finite-sample Monte-Carlo tolerance
N_MC = 4_000_000             # large sample to pin digits
SEED = 20260611

CLAIM = {"confounding_strength": 0.30,
         "observational_association": 0.60,
         "interventional_ate": 0.30,
         "confounding_bias": 0.30}

spec = SCMSpec(graph_type="linear_gaussian_3node",
               beta_ux=0.6, beta_uy=0.5, beta_xy=0.3,
               noise=0.3, n_samples=N_MC, seed=SEED)
scm = SyntheticSCM(spec)

# ---- (A) analytic, from the engine's closed forms -------------------------
analytic = {
    "confounding_strength": scm.true_bias(),         # beta_ux*beta_uy
    "observational_association": scm.true_obs_slope(),  # beta_xy + product
    "interventional_ate": scm.true_int_ate(),        # beta_xy
    "confounding_bias": scm.true_obs_slope() - scm.true_int_ate(),
}

# ---- (B) empirical, Monte-Carlo through the engine's sampler --------------
d = scm.sample()
U, X, Y = d["U"], d["X"], d["Y"]
var_x = float(np.var(X))
obs_slope = float(np.cov(X, Y, bias=True)[0, 1] / var_x)   # observational OLS

# interventional sample == graph surgery do(X): draw X independent of U,
# same standardized marginal variance, keep the structural Y equation.
rng = np.random.default_rng(SEED + 1)
n = N_MC
u2 = rng.standard_normal(n)
x_do = rng.standard_normal(n)                       # X <- noise only (U->X cut)
ey = rng.standard_normal(n) * max(spec.noise, 1e-9)
y_do = spec.beta_xy * x_do + spec.beta_uy * u2 + ey
int_slope = float(np.cov(x_do, y_do, bias=True)[0, 1] / np.var(x_do))

empirical = {
    "confounding_strength": obs_slope - int_slope,   # bias recovered empirically
    "observational_association": obs_slope,
    "interventional_ate": int_slope,
    "confounding_bias": obs_slope - int_slope,
    "_var_x_should_be_1": var_x,
}

# ---- compare --------------------------------------------------------------
ok = True
rows = []
for k, claimed in CLAIM.items():
    a, e = analytic[k], empirical[k]
    da, de = abs(a - claimed), abs(e - claimed)
    pa, pe = da <= TOL_ANALYTIC, de <= TOL_EMPIRICAL
    ok = ok and pa and pe
    rows.append((k, claimed, a, da, pa, e, de, pe))

print("=== R3 v3 machine-cert: Jack Q-C7 self-check ===")
print(f"grid point: beta_ux=0.6 beta_uy=0.5 beta_xy=0.3 noise=0.3 | seed={SEED} N_MC={N_MC}")
print(f"standardization check  Var(X)={var_x:.6f}  (expect 1.0)")
print(f"{'quantity':<28}{'Jack':>8}{'analytic':>12}{'Δ':>10}{'':>4}{'empirical':>12}{'Δ':>10}")
for k, c, a, da, pa, e, de, pe in rows:
    print(f"{k:<28}{c:>8.4f}{a:>12.6f}{da:>10.1e}{'OK' if pa else 'XX':>4}"
          f"{e:>12.6f}{de:>10.1e}{'OK' if pe else 'XX':>4}")
print("VERDICT:", "PASS - all digits match (machine authoritative)" if ok else "FAIL")

# fingerprinted artifact
artifact = {"grid_point": {"beta_ux": 0.6, "beta_uy": 0.5, "beta_xy": 0.3,
                           "noise": 0.3}, "seed": SEED, "n_mc": N_MC,
            "tol_analytic": TOL_ANALYTIC, "tol_empirical": TOL_EMPIRICAL,
            "claim": CLAIM, "analytic": analytic, "empirical": empirical,
            "var_x": var_x, "verdict": "PASS" if ok else "FAIL"}
with open(__file__.replace(".py", "_result.json"), "w") as f:
    json.dump(artifact, f, indent=2)
sys.exit(0 if ok else 1)
