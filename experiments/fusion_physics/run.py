"""Fusion Phase C · L0 physical-constraint dynamics on the spine.

  • Symplectic integrator (velocity-Verlet) conserves energy: |ΔH| ~ 1e-5 over 10000
    steps, vs explicit Euler which injects energy and grows ~170% (orders worse).
  • Energy monitor flags drift beyond threshold (non-fatal).
  • PhysicsLayer ANSWERs a certified-stable evolution (drift < 1e-3, regime declared
    honestly) and ABSTAINS on a drifting one — the physics gate never passes what it
    cannot certify.

Run:  .venv/bin/python experiments/fusion_physics/run.py
"""
from __future__ import annotations
import numpy as np

from theone.layer0_physics import SymplecticIntegrator, ExplicitEuler, EnergyMonitor, PhysicsLayer


def main():
    print("=== Fusion Phase C: L0 physical-constraint dynamics (symplectic gate) ===\n")
    ok = True

    si = SymplecticIntegrator(omega=1.0, dt=0.01)
    ee = ExplicitEuler(omega=1.0, dt=0.01)
    Es, _ = si.evolve([1.0], [0.0], 10000)
    Ee, _ = ee.evolve([1.0], [0.0], 10000)
    sd = float(Es.max() - Es.min())
    ed = float(abs(Ee[-1] - Ee[0]))
    print(f"symplectic (velocity-Verlet): energy drift over 10000 steps = {sd:.2e}  (<1e-3)")
    print(f"explicit Euler (foil):        energy grew {Ee[0]:.3f} -> {Ee[-1]:.3f}  "
          f"(drift {ed:.2e}, ~{ed/sd:.0e}x worse, unbounded)")
    cons_ok = sd < 1e-3 and ed > 100 * sd
    ok &= cons_ok

    mon = EnergyMonitor(threshold=1e-3)
    print(f"\nenergy monitor: symplectic exceeded={mon.check(Es)['exceeded']} | "
          f"explicit exceeded={mon.check(Ee)['exceeded']} (alerts logged={len(mon.alerts)})")
    ok &= (not mon.check(Es)["exceeded"]) and mon.check(Ee)["exceeded"]

    print("\nPhysicsLayer on the spine:")
    L0 = PhysicsLayer(omega=1.0, dt=0.01)
    v = L0.run({"q0": [1.0], "p0": [0.0], "steps": 10000})
    if v.is_answer():
        _, info = v.credential.verify()
        print(f"  stable SHO evolution -> ANSWER | drift={v.credential.value:.2e} | "
              f"regime='{v.credential.regime}' | recompute gap={info.get('gap', 0):.1e}")
    answer_ok = v.is_answer() and v.credential.value < 1e-3

    # a large-dt evolution that drifts past tolerance -> ABSTAIN
    L0_bad = PhysicsLayer(omega=5.0, dt=0.2)
    vb = L0_bad.run({"q0": [1.0], "p0": [0.0], "steps": 10000})
    abstain_ok = not vb.is_answer()
    print(f"  coarse dt (ω=5, dt=0.2) -> {'ABSTAIN' if abstain_ok else 'ANSWER'}: "
          f"{vb.reason if abstain_ok else 'drift=%.2e' % vb.credential.value}")
    ok &= answer_ok and abstain_ok

    print("\nL0 contract (worldview bold, verification honest): the symplectic/energy math is")
    print("real and tested; the credential certifies energy conservation ONLY within the declared")
    print("regime (latent with Hamiltonian structure). The gate passes a certified-stable")
    print("evolution and abstains on a drifting one — never a claim beyond what the test shows.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
