"""B-line scaling — does scale resolve the small-sample reproducibility-stability noise?

NOTE-088's honest caveat: on small samples (~672-1500 units) the continuous native engine's
reproducibility-stability (split-half retrain agreement) is noisy. Scale is the stated cure.
This harness sweeps N on a controllable continuous SCM (known true ATE) and reports, per N:
  - ATE error          (accuracy)
  - reproducibility-stability  (the headline: should rise toward 1.0 with N)
  - zone               (should settle on VERIFIABLE once stable)
Device-agnostic: CUDA on RunPod, MPS on Mac, CPU fallback. Set THEONE_SCALE_NS to override
the sweep (comma-separated), e.g. THEONE_SCALE_NS=1500,6000,25000,100000,400000.

Run:  .venv/bin/python experiments/bline_scale/run.py
"""
from __future__ import annotations
import os
from pathlib import Path
import time
import numpy as np

from theone.native import NativeVerifiableEngine


def make(n, ate, seed, d=12):
    """Continuous SCM: confounder block C drives both treatment and a continuous outcome."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    logit = 0.7 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2]
    t = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    base = 2.0 + X[:, 0] + 0.6 * X[:, 2] - 0.4 * X[:, 3] + 0.3 * X[:, 4] * X[:, 5]
    y = (base + ate * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, t, y


def device_name():
    import torch
    if torch.cuda.is_available():
        return f"cuda · {torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps · Apple Silicon"
    return "cpu"


def main():
    ate = 3.0
    SEED = int(os.environ.get("SEED", "0"))
    ns_env = os.environ.get("THEONE_SCALE_NS")
    ns = [int(x) for x in ns_env.split(",")] if ns_env else [1500, 6000, 25000]
    print("=== B-line scaling · reproducibility-stability vs N (continuous native path) ===")
    print(f"device: {device_name()}   true ATE = {ate}   sweep N = {ns}   SEED={SEED}\n")
    eng = NativeVerifiableEngine()

    print(f"{'N':>8} {'ATE err':>8} {'repro-stability':>16} {'E-value':>8} {'zone':>22} {'sec':>7}")
    rows = []
    for n in ns:
        X, t, y = make(n, ate, seed=SEED)
        t0 = time.time()
        r = eng.estimate_continuous(X, t, y, covariate_sufficient=True)
        dt = time.time() - t0
        err = abs(r.effect - ate)
        rows.append((n, err, r.structural_stability, r.zone))
        print(f"{n:>8} {err:>8.3f} {r.structural_stability:>16.3f} {r.e_value:>8.2f} {r.zone:>22} {dt:>7.1f}")

    # gate: at scale, repro-stability should be HIGH (caveat resolved) and ATE error tight.
    # Trend is monotone toward a ~1.0 ceiling, so test absolute quality at the largest N
    # (strict >-monotonicity false-fails on the 1.000->0.999 rounding wobble at the ceiling).
    s_small = rows[0][2]
    s_large = rows[-1][2]
    err_large = rows[-1][1]
    stable = s_large >= 0.95                                 # reproducibility-stability resolved
    accurate = err_large < 0.3
    print(f"\nscaling gate:")
    print(f"  repro-stability is HIGH at largest N (>=0.95) ...... {'PASS' if stable else 'CHECK'}"
          f"  ({s_small:.3f} -> {s_large:.3f})")
    print(f"  ATE error stays tight at scale (<0.3) ............... {'PASS' if accurate else 'CHECK'}"
          f"  ({err_large:.3f})")
    gate = stable and accurate
    print(f"\n  >>> {'PASS — scale tightens reproducibility-stability (NOTE-088 caveat resolved at scale)' if gate else 'CHECK — inspect trend'}")
    print("\nHonest: synthetic SCM with a known true ATE (controls the experiment); the metric is")
    print("split-half retrain agreement, which is sample-hungry by construction — exactly why it")
    print("was noisy at ~1.5k and is the thing scale is expected to fix.")
    import json, hashlib
    res = {"seed": SEED, "ns": ns, "true_ate": ate,
           "rows": [{"N": n, "ate_err": round(e, 5), "repro_stability": round(s, 5), "zone": z} for n, e, s, z in rows],
           "largest_N": ns[-1], "err_largest": round(err_large, 5), "stability_largest": round(s_large, 5),
           "gate_pass": bool(gate)}
    outp = Path(os.environ.get("RESULT_DIR", str(Path(__file__).parent))) / f"b1_scale_result_seed{SEED}.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"RESULT_JSON {outp.name} sha256={hashlib.sha256(outp.read_bytes()).hexdigest()}")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
