"""Fusion deepening⑤ · L4 memory legs — pattern recognition + conflict arbitration on
de-confounded causal signatures (not surface text).

  • PatternRecognizer: finds causal edges that RECUR across memories.
  • ConflictArbitrator: flags two live memories that answer the SAME causal question
    (treatment->target | adjustment | regime) with contradictory effects, and proposes a
    resolution — while NOT flagging a same-text-different-regime look-alike.

Run:  .venv/bin/python experiments/fusion_memory_legs/run.py
"""
from __future__ import annotations
from theone.memory.sovereign import SovereignMemory
from theone.layer4_memory import PatternRecognizer, ConflictArbitrator


def cred(t, y, adj, eff, regime):
    return {"treatment": t, "target": y, "adjustment_set": adj, "effect": eff, "regime": regime}


def main():
    print("=== Fusion deepening⑤: L4 memory legs (pattern + conflict on signatures) ===\n")
    mem = SovereignMemory(":memory:")
    # recurring X->Y, plus a genuine conflict (same question, different effect), plus a
    # regime look-alike (NOT a conflict), plus a separate recurring A->B.
    mem.remember("X drives Y (study 1)", cred("X", "Y", ["U"], 0.66, "normal"), "src1")
    mem.remember("X->Y, different source", cred("X", "Y", ["U"], 0.45, "normal"), "src2")   # conflict
    mem.remember("X->Y under stress", cred("X", "Y", ["U"], 0.30, "stressed"), "src3")       # not a conflict
    mem.remember("A affects B", cred("A", "B", [], 0.50, "normal"), "src4")
    mem.remember("A->B replication", cred("A", "B", [], 0.52, "normal"), "src5")             # within tol
    ok = True

    pr = PatternRecognizer(mem)
    edges = pr.frequent_edges(min_support=2)
    print("frequent causal edges (support >= 2):")
    for e in edges:
        print(f"   {e['edge']}  count={e['count']}  support={e['support']}")
    edge_map = {tuple(e["edge"]): e["count"] for e in edges}
    ok &= edge_map.get(("X", "Y")) == 3 and edge_map.get(("A", "B")) == 2

    ca = ConflictArbitrator(mem)
    conflicts = ca.find_conflicts(effect_tol=0.1)
    print(f"\nconflicts detected: {len(conflicts)}")
    for c in conflicts:
        print(f"   question: {c['question']}")
        print(f"   effect spread={c['effect_spread']} among {[m['effect'] for m in c['members']]} "
              f"-> resolution: {c['resolution']}")
    # exactly one conflict: the X->Y|adj=[U]|regime=normal pair (0.66 vs 0.45)
    one = len(conflicts) == 1 and conflicts[0]["question"] == "X->Y|adj=[U]|regime=normal"
    ok &= one
    # the stressed-regime memory must NOT be merged into the conflict (different question)
    ok &= all("stressed" not in c["question"] for c in conflicts)
    # the A->B pair (0.50 vs 0.52) is within tol -> not a conflict
    ok &= all(not c["question"].startswith("A->B") for c in conflicts)
    mem.close()

    print("\nL4-legs contract: recurrence and conflict are judged on the de-confounded causal")
    print("signature, so a genuine contradiction (same question, different effect) is caught while")
    print("a regime/text look-alike is not; resolutions are PROPOSED (sovereignty: never silent edits).")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
