"""The One's decision闭环 with TWO independent gates: recomputability AND
admissibility. The os_loop already abstains unless an independent recompute (pgmpy)
matches the engine. Here we add the constraint credential (§ auditable criterion 4
extended): even a pgmpy-verified answer is downgraded to ABSTAIN if it violates a
*declared domain constraint* (bounds, effect-sign monotonicity).

The decisive case: a MISSPECIFIED model whose causal effect is sign-flipped. The
engine computes it exactly, so pgmpy verifies it (both agree on the wrong structure) —
a bare 'computation-exact + recomputable' credential would emit it. The constraint
credential catches that it contradicts the declared domain knowledge (effect should be
positive) and abstains. Two orthogonal gates: 'did we compute it right' AND 'is the
result admissible given what we declared about the world'.

Run:  .venv/bin/python experiments/os_loop_constrained/run.py
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_cc = importlib.util.spec_from_file_location("cc", HERE.parent / "constraint_credential" / "run.py")
CC = importlib.util.module_from_spec(_cc); _cc.loader.exec_module(CC)
_os = importlib.util.spec_from_file_location("osl", HERE.parent / "os_loop" / "run.py")
OS = importlib.util.module_from_spec(_os); _os.loader.exec_module(OS)


def the_one_constrained(g, declared_sign="positive", regime="normal"):
    """One credentialed pass with both gates."""
    eng = InterventionEngine(g)
    do1 = round(eng.query_intervention("Y", 1, {"X": 1}).value, 6)
    do0 = round(eng.query_intervention("Y", 1, {"X": 0}).value, 6)
    comp1 = round(eng.query_intervention("Y", 0, {"X": 1}).value, 6)
    # Gate 1: independent recomputability (pgmpy true do on full model)
    try:
        truth = OS.pgmpy_true_do(g, "X", "Y")
        recomputable = abs(do1 - truth) < 1e-6
    except Exception:
        truth, recomputable = None, False
    # Gate 2: admissibility (constraint credential)
    checks = CC.constraint_credential(do1, do0=do0, declared_sign=declared_sign, complement=comp1)
    admissible = all(v == "PASS" for v in checks.values() if v != "NA")
    # decision needs BOTH
    if not recomputable:
        decision, reason = "ABSTAIN", "independent recompute mismatch"
    elif not admissible:
        decision, reason = "ABSTAIN", f"violates declared constraint: {[k for k,v in checks.items() if v=='VIOLATED']}"
    else:
        decision, reason = "ANSWER", "recomputable AND admissible"
    return {"do_x1": do1, "do_x0": do0, "pgmpy_true": round(truth, 6) if truth is not None else None,
            "recomputable": recomputable, "constraint_checks": checks,
            "admissible": admissible, "decision": decision, "reason": reason}


def main():
    print("=== the_one with two orthogonal gates: recomputable AND admissible ===\n")

    # (A) normal model, declared positive effect — both gates pass
    g = CC.confounded(3, 7)
    a = the_one_constrained(g, declared_sign="positive")
    print(f"(A) normal model (declared positive): do(X=1)={a['do_x1']} do(X=0)={a['do_x0']}")
    print(f"    gate1 recomputable(pgmpy)={a['recomputable']} | gate2 admissible={a['admissible']} {a['constraint_checks']}")
    print(f"    → DECISION: {a['decision']} ({a['reason']})\n")

    # (B) MISSPECIFIED model (effect sign flipped) but still declared positive
    gf = CC.confounded(3, 7, flip=True)
    b = the_one_constrained(gf, declared_sign="positive")
    print(f"(B) misspecified model (sign flipped, still declared positive): do(X=1)={b['do_x1']} do(X=0)={b['do_x0']}")
    print(f"    gate1 recomputable(pgmpy)={b['recomputable']}  ← pgmpy VERIFIES it (engine exact on the wrong structure)")
    print(f"    gate2 admissible={b['admissible']} {b['constraint_checks']}  ← constraint CATCHES the sign violation")
    print(f"    → DECISION: {b['decision']} ({b['reason']})\n")

    ok = (a["decision"] == "ANSWER" and b["decision"] == "ABSTAIN" and b["recomputable"])
    print("RESULT:", "PASS — the misspecified model is pgmpy-verified yet abstained on by the "
          "constraint gate; the two gates are orthogonal and the credential is strictly stronger."
          if ok else "unexpected")
    print("\nThe One's metacognitive decision now rests on TWO independent, third-party-"
          "recomputable checks: 'computed exactly' (pgmpy) and 'admissible given declared "
          "domain knowledge' (constraint) — the latter catches misspecification the former passes.")


if __name__ == "__main__":
    main()
