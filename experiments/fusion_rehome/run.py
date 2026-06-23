"""Fusion Phase A · step 1 — re-home the 3 VERIFIED subsystems onto the spine, prove
regression (<1e-6) and that the spine interface preserves their behavior.

L2 causal  : wraps the frozen InterventionEngine; credential recomputed by pgmpy IPRG.
L4 memory  : wraps SovereignMemory; signature-indexed recall, credential re-read from store.
L5 execute : wraps SafeExecutor; EXECUTE→ANSWER, BLOCK/ABSTAIN→ABSTAIN.

Regression contract: the wrapped layer's numbers must equal the underlying verified
module's numbers (engine do == layer do == pgmpy IPRG, all to <1e-6), and the honest
behaviors (do≠obs, confounded look-alike rejected, dangerous command blocked) must survive.

Run:  .venv/bin/python experiments/fusion_rehome/run.py
"""
from __future__ import annotations
import numpy as np

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.core.spine import Spine, Decision
from theone.layer2_world_model.causal_layer import CausalLayer
from theone.layer2_world_model.iprg import pgmpy_do1
from theone.layer4_memory.memory_layer import MemoryLayer
from theone.layer5_execution.execution_layer import ExecutionLayer
from theone.memory.signature import CausalSignature


def confounded_fork() -> CausalGraph:
    """U confounds X and Y;  U->X, U->Y, X->Y.  do(X=1) != P(Y=1|X=1)."""
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    # Y parents sorted = (U, X)
    g.set_cpt("Y", {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
                    (1, 0): {0: 0.4, 1: 0.6}, (1, 1): {0: 0.1, 1: 0.9}})
    return g


def main():
    print("=== Fusion Phase A·1: re-homing the 3 verified subsystems onto the spine ===\n")
    ok = True
    g = confounded_fork()

    # ---- L2 regression: layer do == engine do == pgmpy IPRG -----------------
    eng = InterventionEngine(g)
    engine_do = float(eng.query_intervention("Y", 1, {"X": 1}).value)
    obs = float(eng.query_observation("Y", 1, {"X": 1}).value)
    v = CausalLayer().run({"graph": g, "treatment": "X", "target": "Y"})
    layer_do = v.value["do_value"] if v.is_answer() else None
    iprg = pgmpy_do1(g, "X", "Y")
    reg_gap = abs(layer_do - engine_do) if layer_do is not None else float("nan")
    iprg_gap = abs((layer_do if layer_do is not None else 0) - iprg)
    print("L2 causal (re-homed InterventionEngine):")
    print(f"  layer do(X=1)={layer_do:.6f}  engine do={engine_do:.6f}  pgmpy IPRG={iprg:.6f}")
    print(f"  regression gap layer-vs-engine={reg_gap:.2e}  |  IPRG gap={iprg_gap:.2e}  "
          f"| obs P(Y|X=1)={obs:.6f}  do-obs={layer_do-obs:+.4f} (causal != correlational)")
    l2_ok = v.is_answer() and reg_gap < 1e-9 and iprg_gap < 1e-6 and abs(layer_do - obs) > 1e-3
    print(f"  -> {'ANSWER' if v.is_answer() else 'ABSTAIN'} | regression {'PASS' if l2_ok else 'FAIL'}")
    ok &= l2_ok

    # ---- L4 regression: signature recall ANSWERs; confounded look-alike ABSTAINs ----
    print("\nL4 memory (re-homed SovereignMemory):")
    mem = MemoryLayer(":memory:")
    cred_normal = {"treatment": "X", "target": "Y", "adjustment_set": ["U"],
                   "effect": round(layer_do, 6), "regime": "normal"}
    mem.run({"op": "remember", "text": "X->Y under normal regime", "credential": cred_normal,
             "source": "fusion_rehome"})
    # a memory that READS similar but has a different regime/effect (the trap)
    cred_trap = {"treatment": "X", "target": "Y", "adjustment_set": ["U"],
                 "effect": 0.20, "regime": "stressed"}
    mem.run({"op": "remember", "text": "X->Y looks the same", "credential": cred_trap,
             "source": "fusion_rehome"})
    q_match = CausalSignature.from_credential(cred_normal)
    rv = mem.run({"op": "recall", "query_signature": q_match, "match_threshold": 0.5})
    hit_ok = rv.is_answer() and rv.value["recall"].signature.regime == "normal"
    # query for a regime we never stored well → structural mismatch → ABSTAIN
    q_miss = CausalSignature(treatment="X", target="Y", adjustment_set=("U",),
                             effect=0.66, regime="crisis")
    rv_miss = mem.run({"op": "recall", "query_signature": q_miss, "match_threshold": 0.5})
    miss_ok = (not rv_miss.is_answer())
    print(f"  exact-signature recall -> {'ANSWER' if rv.is_answer() else 'ABSTAIN'} "
          f"(got regime={rv.value['recall'].signature.regime if rv.is_answer() else 'n/a'}) "
          f"{'PASS' if hit_ok else 'FAIL'}")
    print(f"  confounded look-alike (unseen regime) -> "
          f"{'ABSTAIN' if not rv_miss.is_answer() else 'ANSWER'} {'PASS' if miss_ok else 'FAIL'}")
    mem.close()
    ok &= hit_ok and miss_ok

    # ---- L5 regression: safe ANSWERs, dangerous/inadmissible ABSTAIN --------
    print("\nL5 execution (re-homed SafeExecutor):")
    ex = ExecutionLayer(sandbox_root="/tmp")
    safe = ex.run({"action_kind": "command", "command": "echo hello"})
    danger = ex.run({"action_kind": "command", "command": "rm -rf /"})
    # causal-driven action whose credential is NOT independently-recomputable -> ABSTAIN
    bad_causal = ex.run({"action_kind": "write", "target": "/tmp/out.txt", "content": "x",
                         "causal_credential": {"recomputable": False, "admissible": True}})
    good_causal = ex.run({"action_kind": "write", "target": "/tmp/out.txt", "content": "x",
                          "causal_credential": {"recomputable": True, "admissible": True}})
    l5_ok = (safe.is_answer() and not danger.is_answer()
             and not bad_causal.is_answer() and good_causal.is_answer())
    print(f"  'echo hello'        -> {'ANSWER' if safe.is_answer() else 'ABSTAIN'}")
    print(f"  'rm -rf /'          -> {'ABSTAIN' if not danger.is_answer() else 'ANSWER'} ({danger.reason})")
    print(f"  causal, unverified  -> {'ABSTAIN' if not bad_causal.is_answer() else 'ANSWER'} ({bad_causal.reason})")
    print(f"  causal, verified    -> {'ANSWER' if good_causal.is_answer() else 'ABSTAIN'}")
    print(f"  -> {'PASS' if l5_ok else 'FAIL'}")
    ok &= l5_ok

    print("\nRe-homing contract:")
    print("  • L2 wrapped do == frozen engine do (regression <1e-9) AND == pgmpy IPRG (<1e-6)")
    print("  • L4 signature recall preserves confounding-immunity (trap look-alike rejected)")
    print("  • L5 preserves sandbox + two-gate causal admissibility")
    print("  The 3 verified subsystems now speak the spine's LayerVerdict — behavior unchanged.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
