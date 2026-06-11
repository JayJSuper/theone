"""CLI. CC-12. Commands: theone demo causal | theone test | theone bench mvp2a"""
from __future__ import annotations
import argparse
import json
import sys
import tempfile


def _demo_causal() -> int:
    from .types import Variable, TheOneConfig
    from .causal.graph import CausalGraph
    from .causal.engine import InterventionEngine
    from .memory.store import MemoryStore
    from .agent.orchestrator import Orchestrator

    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("X", {(1,): {1: 0.8, 0: 0.2}, (0,): {1: 0.2, 0: 0.8}})
    g.set_cpt("Y", {(1, 1): {1: 0.9, 0: 0.1}, (0, 1): {1: 0.5, 0: 0.5},
                    (1, 0): {1: 0.6, 0: 0.4}, (0, 0): {1: 0.2, 0: 0.8}})
    eng = InterventionEngine(g)
    cmp_ = eng.compare("X", "Y", TheOneConfig())
    print("=== The One demo: frozen three-node confounded graph (T1) ===")
    print(f"P(Y=1|X=1)      = {eng.query_observation(chr(89),1,{chr(88):1}).value:.6f}   (frozen truth 0.82)")
    print(f"P(Y=1|do(X=1))  = {eng.query_intervention(chr(89),1,{chr(88):1}).value:.6f}   (frozen truth 0.70)")
    print(f"obs ATE = {cmp_.obs_ate:.6f} (0.54) | int ATE = {cmp_.int_ate:.6f} (0.30)")
    print(f"are_different   = {cmp_.are_different}")
    with tempfile.TemporaryDirectory() as td:
        agent = Orchestrator(eng, MemoryStore(td + "/m.db"))
        resp = agent.handle("P(Y=1|do(X=1))")
        print("--- credential ---")
        print(json.dumps(resp.credential, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="theone")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("demo"); d.add_argument("what", choices=["causal"])
    sub.add_parser("test")
    b = sub.add_parser("bench"); b.add_argument("which", choices=["mvp2a"])
    b.add_argument("--phase", choices=["calibrate", "frozen"], required=True)
    b.add_argument("--out", default="experiments/mvp2a")
    b.add_argument("--grid-config", default=None)
    b.add_argument("--frozen-config", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "demo":
        return _demo_causal()
    if args.cmd == "test":
        import pytest
        return pytest.main(["-q"])
    if args.cmd == "bench":
        from .bench.runner import run_calibration, run_frozen
        if args.phase == "calibrate":
            cfg = json.load(open(args.grid_config)) if args.grid_config else {
                "beta_ux": [0.3, 0.6], "beta_uy": [0.3, 0.5],
                "beta_xy": [0.2, 0.4], "noise": [0.5], "n_samples": 2000,
                "base_seed": 42}  # toy grid: pipeline check only; real grid = Q-C7
            rep = run_calibration(cfg, args.out)
        else:
            if not args.frozen_config:
                print("frozen phase requires --frozen-config", file=sys.stderr)
                return 2
            rep = run_frozen(json.load(open(args.frozen_config)), args.out)
        print(json.dumps(rep, indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
