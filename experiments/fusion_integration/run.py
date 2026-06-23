"""Fusion Phase E · the full 6-layer credential spine, end-to-end.

All six layers (L0 physics, L1 perception, L2 causal, L3 decision, L4 memory, L5
execution) run in topological order through one Spine over a shared context. The
system ANSWERs only if every layer passes BOTH gates (admissible AND its credential
independently recomputes); a fault anywhere trips the abstain bus at that layer.

This is the assembled "super form": 6 independently-verifiable layers threaded by one
spine — perceive (L1) on a physically-admissible substrate (L0), reason causally (L2),
decide by active inference (L3), recall sovereign memory (L4), and act under audit (L5),
each emitting a third-party-recomputable credential or abstaining.

Run:  .venv/bin/python experiments/fusion_integration/run.py
"""
from __future__ import annotations
import numpy as np

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.core.spine import Spine, Decision
from theone.layer0_physics import PhysicsLayer
from theone.layer1_perception import PerceptionLayer
from theone.layer2_world_model import CausalLayer
from theone.layer3_decision import DecisionLayer
from theone.layer4_memory import MemoryLayer
from theone.layer5_execution import ExecutionLayer
from theone.memory.signature import CausalSignature


def confounded_fork():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("X", {(0,): {0: 0.8, 1: 0.2}, (1,): {0: 0.3, 1: 0.7}})
    g.set_cpt("Y", {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
                    (1, 0): {0: 0.4, 1: 0.6}, (1, 1): {0: 0.1, 1: 0.9}})
    return g


def vfe_problem(seed=0, d_o=8, d_z=4):
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_o, d_z))
    o = W @ (rng.standard_normal(d_z) * 0.3)
    return W, o / (np.linalg.norm(o) + 1e-9) * np.sqrt(2.0)


def build_context(mem_layer):
    g = confounded_fork()
    W, o = vfe_problem()
    t = np.linspace(0, 10, 1000)
    # pre-store a memory whose signature matches the causal effect we will query
    do_val = 0.66
    cred = {"treatment": "X", "target": "Y", "adjustment_set": ["U"],
            "effect": do_val, "regime": "normal"}
    mem_layer.run({"op": "remember", "text": "X->Y normal regime", "credential": cred})
    q_sig = CausalSignature.from_credential(cred)
    return {
        "q0": [1.0], "p0": [0.0], "steps": 10000,            # L0
        "signal": np.sin(2 * np.pi * 1.0 * t),               # L1
        "graph": g, "treatment": "X", "target": "Y",          # L2
        "W": W, "observation": o,                             # L3
        "op": "recall", "query_signature": q_sig, "match_threshold": 0.5,  # L4
        "action_kind": "command", "command": "echo integrated",            # L5
    }


def run_and_report(label, layers, ctx):
    sv = Spine(layers).run(ctx)
    if sv.is_answer():
        order = " -> ".join(c.layer for c in sv.credentials)
        gaps = []
        for c in sv.credentials:
            _, info = c.verify()
            gaps.append(info.get("gap", 0.0))
        print(f"  {label:<22} SYSTEM ANSWER | {len(sv.credentials)}/6 layers | "
              f"max recompute gap={max(gaps):.1e}")
        print(f"  {'':<22} stacked credential: {order}")
    else:
        print(f"  {label:<22} SYSTEM ABSTAIN @ {sv.abstained_at}: {sv.reason}")
    return sv


def main():
    print("=== Fusion Phase E: the full 6-layer credential spine, end-to-end ===\n")
    L4 = MemoryLayer(":memory:")
    layers = [PhysicsLayer(), PerceptionLayer(hidden_dim=64, seed=1),
              CausalLayer(), DecisionLayer(), L4, ExecutionLayer(sandbox_root="/tmp")]

    print("Scenario A — healthy: all six layers admissible and recomputable:")
    ctx = build_context(L4)
    a = run_and_report("healthy", layers, ctx)

    print("\nScenario B — fault injected at L4 (query an unseen causal regime):")
    ctx_b = dict(ctx)
    ctx_b["query_signature"] = CausalSignature(treatment="X", target="Y",
                                               adjustment_set=("U",), effect=0.66, regime="crisis")
    b = run_and_report("L4 unseen regime", layers, ctx_b)

    print("\nScenario C — fault injected at L5 (dangerous command):")
    ctx_c = dict(ctx); ctx_c["command"] = "rm -rf /"
    c = run_and_report("L5 dangerous cmd", layers, ctx_c)

    ok = (a.is_answer() and not b.is_answer() and b.abstained_at == "L4_memory"
          and not c.is_answer() and c.abstained_at == "L5_execution")
    L4.close()
    print("\nThe assembled super-form: 6 independently-verifiable layers, one spine.")
    print("  • a system answer is a STACKED, end-to-end recomputable receipt across L0..L5")
    print("  • any single layer's admissibility OR recomputability failure trips the abstain bus")
    print("  • no confident-narrow-wrong output can traverse all six gates")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
