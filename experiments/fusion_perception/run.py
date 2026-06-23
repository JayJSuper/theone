"""Fusion Phase B · L1 continuous perception on the spine.

Demonstrates the new, genuinely-additive layer:
  • SSM encoder: a 1Hz sine (100Hz, 10s = 1000 steps) → stable latent (spectral
    radius < 1), reconstructed by a LINEAR decoder to MSE < 1e-3 → ANSWER.
  • Information-preserving: a near-lossless linear encoder reconstructs even white
    noise (correct — a faithful encoder loses nothing; this is not a failure).
  • Honest ABSTAIN: an unstable encoder (spectral radius >= 1) and a degenerate
    (NaN) input are both refused rather than emitting a bad latent.
  • Temporal lock: strictly monotonic nanosecond stamps; a stale stamp is rejected.
  • Modality registry: register/resolve in O(1); unknown modality raises.
The latent H is passed forward as the L1->L2 connector (the seam the
native_causal_latent probes prototyped: continuous latent -> verifiable do).

Run:  .venv/bin/python experiments/fusion_perception/run.py
"""
from __future__ import annotations
import numpy as np

from theone.core.spine import Decision
from theone.layer1_perception import (
    PerceptionLayer, TemporalLock, TemporalConflictError,
    ModalityRegistry, ModalityConfig, UnknownModalityError,
)


def main():
    print("=== Fusion Phase B: L1 continuous perception (SSM encoder on the spine) ===\n")
    ok = True

    # ---- SSM encoder: faithful on structured signal --------------------------
    t = np.linspace(0, 10, 1000)
    sine = np.sin(2 * np.pi * 1.0 * t)
    L1 = PerceptionLayer(hidden_dim=64, spectral_radius=0.9, seed=1)
    v = L1.run({"signal": sine})
    if v.is_answer():
        _, info = v.credential.verify()
        H = v.value["latent"]
        print(f"sine 1Hz/100Hz/10s -> ANSWER | spectral_radius="
              f"{v.credential.evidence['spectral_radius']:.4f} (<1, stable) | "
              f"recon MSE={v.credential.value:.2e} (<1e-3) | latent H shape={H.shape} | "
              f"recompute gap={info.get('gap', 0):.1e}")
        sine_ok = v.credential.value < 1e-3 and v.credential.evidence["spectral_radius"] < 1.0
    else:
        print(f"sine -> ABSTAIN: {v.reason}")
        sine_ok = False
    ok &= sine_ok

    # ---- information-preserving: lossless encoder reconstructs noise too ------
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(1000)
    vn = L1.run({"signal": noise})
    noise_ok = vn.is_answer() and vn.credential.value < 1e-3   # faithful, as it should be
    print(f"white noise        -> {'ANSWER' if vn.is_answer() else 'ABSTAIN'} "
          f"(MSE={vn.credential.value:.2e}) — near-lossless encoder loses nothing (correct)")
    ok &= noise_ok

    # ---- honest ABSTAIN: unstable encoder + degenerate input -----------------
    v_unstable = PerceptionLayer(spectral_radius=1.0).run({"signal": sine})
    nan_sig = sine.copy(); nan_sig[10] = np.nan
    v_nan = L1.run({"signal": nan_sig})
    abstain_ok = (not v_unstable.is_answer()) and (not v_nan.is_answer())
    print(f"unstable (ρ=1.0)   -> {'ABSTAIN' if not v_unstable.is_answer() else 'ANSWER'}: {v_unstable.reason}")
    print(f"NaN in signal      -> {'ABSTAIN' if not v_nan.is_answer() else 'ANSWER'}: {v_nan.reason}")
    ok &= abstain_ok

    # ---- temporal lock -------------------------------------------------------
    lock = TemporalLock()
    stamps = [lock.stamp() for _ in range(1000)]
    monotonic = all(b > a for a, b in zip(stamps, stamps[1:]))
    try:
        lock.accept(stamps[-1])           # stale -> must be rejected
        conflict_caught = False
    except TemporalConflictError:
        conflict_caught = True
    print(f"temporal lock      -> 1000 stamps strictly increasing={monotonic}; "
          f"stale stamp rejected={conflict_caught} (conflicts={lock.conflicts})")
    ok &= monotonic and conflict_caught

    # ---- modality registry ---------------------------------------------------
    reg = ModalityRegistry()
    for nm, dim, hz in [("optical", 3, 60.0), ("acoustic", 1, 44100.0),
                        ("force", 6, 1000.0), ("em", 2, 1e6), ("gravity", 1, 10.0)]:
        reg.register(ModalityConfig(nm, dim, hz))
    got = reg.get("acoustic")
    try:
        reg.get("telepathy")
        unknown_raised = False
    except UnknownModalityError:
        unknown_raised = True
    print(f"modality registry  -> {len(reg.names)} registered; resolve 'acoustic' dim="
          f"{got.input_dim}; unknown raises={unknown_raised}")
    ok &= (len(reg.names) == 5 and unknown_raised)

    print("\nL1 contract: continuous signal -> stable latent (spectral radius < 1) -> faithful")
    print("linear reconstruction < 1e-3 (information-preserving), with honest ABSTAIN on an")
    print("unstable encoder or degenerate input; strict temporal order; O(1) modality routing.")
    print("Latent H is emitted as the L1->L2 connector (native_causal_latent probes).")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
