"""End-to-end: causal effect → two orthogonal gates → auditable safe execution.

Demonstrates The One's execution boundary: an action justified by a causal
recommendation reaches the world ONLY if that recommendation is independently-
recomputable (pgmpy) AND admissible (constraint credential), AND the action itself
passes sandbox containment / command denylist. Every step emits an auditable record.

This answers the "Part 3 / execution layer" question the right way: not a sandbox on
top of a random world-model, but a sandbox gated on *verifiable causal advice*.

Run:  .venv/bin/python experiments/safe_execution/run.py
"""
from __future__ import annotations
import importlib.util, json, shutil
from pathlib import Path
from theone.execution import SafeExecutor

HERE = Path(__file__).parent
SANDBOX = HERE / "sandbox"
_oc = importlib.util.spec_from_file_location("osc", HERE.parent / "os_loop_constrained" / "run.py")
OC = importlib.util.module_from_spec(_oc); _oc.loader.exec_module(OC)


def causal_credential(g, declared_sign="positive"):
    """Run the two-gate causal decision and package it for the executor."""
    r = OC.the_one_constrained(g, declared_sign=declared_sign)
    return {"recomputable": r["recomputable"], "admissible": r["admissible"],
            "do_x1": r["do_x1"], "decision": r["decision"]}, r


def main():
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)
    ex = SafeExecutor(sandbox_root=str(SANDBOX))
    print(f"=== auditable safe execution (sandbox={SANDBOX.name}) ===\n")

    # (1) non-causal housekeeping write, safe path -> EXECUTE -> real run
    c1 = ex.propose_write(str(SANDBOX / "log.txt"), "run started\n", provenance="housekeeping")
    print(f"(1) housekeeping write: decision={c1.decision} | {c1.dry_run_output}")
    print(f"    execute -> {ex.execute(c1, content='run started\\n', confirm=True)}\n")

    # (2) dangerous command -> BLOCK (denylist)
    c2 = ex.propose_command("rm -rf /", provenance="adversarial")
    print(f"(2) command 'rm -rf /': decision={c2.decision} ({c2.reason})\n")

    # (3) path escape -> BLOCK (sandbox containment)
    c3 = ex.propose_write("../../../etc/passwd", "x", provenance="adversarial")
    print(f"(3) path escape '../../../etc/passwd': decision={c3.decision} ({c3.reason})\n")

    # (4) CAUSAL-DRIVEN action, recommendation VERIFIED (recomputable AND admissible) -> EXECUTE
    g_ok = OC.CC.confounded(3, 7)                      # normal model, declared positive
    cc_ok, r_ok = causal_credential(g_ok)
    c4 = ex.propose_write(str(SANDBOX / "intervene_ok.txt"),
                          f"act on do(X=1)={cc_ok['do_x1']}\n",
                          causal_credential=cc_ok, provenance="causal:normal-model")
    print(f"(4) causal-driven action (effect={cc_ok['do_x1']}, gate={c4.causal_gate}): "
          f"decision={c4.decision}")
    print(f"    causal driver: recomputable={cc_ok['recomputable']} admissible={cc_ok['admissible']} "
          f"({c4.causal_reason})")
    print(f"    execute -> {ex.execute(c4, content='act\\n', confirm=True)}\n")

    # (5) CAUSAL-DRIVEN action, recommendation MISSPECIFIED (admissible=False) -> ABSTAIN
    g_bad = OC.CC.confounded(3, 7, flip=True)          # sign-flipped, still declared positive
    cc_bad, r_bad = causal_credential(g_bad)
    c5 = ex.propose_write(str(SANDBOX / "intervene_bad.txt"),
                          f"act on do(X=1)={cc_bad['do_x1']}\n",
                          causal_credential=cc_bad, provenance="causal:misspecified-model")
    print(f"(5) causal-driven action on MISSPECIFIED model (effect={cc_bad['do_x1']}, "
          f"gate={c5.causal_gate}): decision={c5.decision}")
    print(f"    causal driver: recomputable={cc_bad['recomputable']} (pgmpy still verifies the "
          f"exact-but-wrong number) admissible={cc_bad['admissible']} ({c5.causal_reason})")
    print(f"    → the action is ABSTAINED on: a pgmpy-verified but inadmissible causal "
          f"recommendation is NOT allowed to touch the world.\n")

    # (6) audit log + sovereignty
    log = ex.audit_log()
    (HERE / "audit_log.json").write_text(json.dumps(log, indent=2, default=str))
    dec = {}
    for c in log:
        dec[c["decision"]] = dec.get(c["decision"], 0) + 1
    print(f"(6) audit log: {len(log)} actions recorded -> {dec}")
    files = sorted(p.name for p in SANDBOX.iterdir())
    print(f"    sandbox now contains: {files}  (intervene_bad.txt absent — abstained)")
    print("\nThe One can now ACT on causal advice, but only advice that is independently-"
          "recomputable AND admissible AND executes within a contained, audited boundary. "
          "Acting is gated on verification, not trust.")


if __name__ == "__main__":
    main()
