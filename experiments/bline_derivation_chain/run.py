"""B5/Q1 · native self-verification via a REPLAYABLE derivation chain — made REAL.

DP proposed (correctly) that native verification without an external oracle = "the
reasoning path can be re-executed". But DP's code was all stubs (every _replay_ returned
the stored output — verifying nothing) and hung "energy conservation" on non-physical
steps (the metaphorical-energy trap). This builds the IDEA for real:

  • every derivation step carries a DETERMINISTIC recompute(inputs)->output;
  • verification RE-EXECUTES each step from its inputs and checks it reproduces the
    recorded output within tolerance (the actual check, not a stub);
  • dependencies are re-evaluated in topological order, so a corrupted intermediate is
    caught downstream;
  • a content hash gives tamper-evidence on top.

No external engine: the chain re-derives itself. This is The One's existing recompute-
credential generalized from one step to a verifiable multi-step chain. Demo: a real
multi-step do() derivation that replays exactly; then we TAMPER a step and the verifier
catches it — and we drop the science-poetry "energy" constraint for non-physical steps.

Run:  .venv/bin/python experiments/bline_derivation_chain/run.py
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class Step:
    sid: str
    op: str
    deps: list                      # ids of steps whose outputs feed this one
    literals: dict                  # literal inputs
    recompute: Callable             # (dep_outputs: dict, literals: dict) -> output
    recorded: Any                   # the output recorded at build time
    tol: float = 1e-9

    def hash(self) -> str:
        c = json.dumps({"sid": self.sid, "op": self.op, "deps": sorted(self.deps),
                        "literals": self.literals, "recorded": _ser(self.recorded)}, sort_keys=True)
        return hashlib.sha256(c.encode()).hexdigest()


def _ser(v):
    return round(v, 12) if isinstance(v, float) else v


class DerivationChain:
    def __init__(self, conclusion_sid: str):
        self.steps: dict[str, Step] = {}
        self.order: list[str] = []
        self.conclusion_sid = conclusion_sid

    def add(self, step: Step):
        self.steps[step.sid] = step
        self.order.append(step.sid)

    def verify(self) -> tuple[bool, list[str]]:
        """RE-EXECUTE every step from its inputs; check it reproduces the recorded output.
        No external oracle — the chain re-derives itself."""
        errors, outputs = [], {}
        for sid in self.order:
            s = self.steps[sid]
            for d in s.deps:
                if d not in outputs:
                    errors.append(f"{sid}: dependency {d} not yet derived"); break
            else:
                try:
                    got = s.recompute({d: outputs[d] for d in s.deps}, s.literals)
                except Exception as e:  # a recompute that throws = not verified (fail-safe)
                    errors.append(f"{sid}: recompute raised {type(e).__name__}"); got = None
                if got is not None:
                    gap = abs(got - s.recorded) if isinstance(got, (int, float)) else (0 if got == s.recorded else 1)
                    if gap > s.tol:
                        errors.append(f"{sid}: replay {got} != recorded {s.recorded} (gap {gap:.2e})")
                    outputs[sid] = got
        ok = not errors
        return ok, errors

    def root_hash(self) -> str:
        return hashlib.sha256("".join(self.steps[s].hash() for s in self.order).encode()).hexdigest()


def build_do_derivation():
    """A real 3-step do(X=1) derivation on a confounded fork: stratum weights -> per-stratum
    P(Y=1|do X=1) -> marginalize. Each step is independently recomputable."""
    pu = 0.4
    yU0, yU1 = 0.45, 0.85       # P(Y=1 | X=1, U=0/1)
    ch = DerivationChain("s3")
    ch.add(Step("s1", "P(U) prior weights", [], {"pu": pu},
                lambda d, l: [1 - l["pu"], l["pu"]], [0.6, 0.4]))
    ch.add(Step("s2", "P(Y=1|do X=1, U)", [], {"yU0": yU0, "yU1": yU1},
                lambda d, l: [l["yU0"], l["yU1"]], [0.45, 0.85]))
    ch.add(Step("s3", "marginalize -> do(X=1)", ["s1", "s2"], {},
                lambda d, l: sum(w * y for w, y in zip(d["s1"], d["s2"])),
                0.6 * 0.45 + 0.4 * 0.85))
    return ch


def main():
    print("=== Q1 made REAL · replayable derivation chain (native self-verification) ===\n")
    ch = build_do_derivation()

    ok, errs = ch.verify()
    print(f"1. honest chain re-executes itself:  verify = {ok}  (conclusion do(X=1) = "
          f"{ch.steps['s3'].recorded})")
    print(f"   root hash (tamper-evidence) = {ch.root_hash()[:16]}…")

    # TAMPER: corrupt the recorded output of an intermediate step
    ch2 = build_do_derivation()
    ch2.steps["s2"].recorded = [0.45, 0.95]      # someone faked a stratum probability
    ok2, errs2 = ch2.verify()
    print(f"\n2. tampered chain (faked s2 output): verify = {ok2}")
    print(f"   caught: {errs2[0] if errs2 else 'NOTHING — would be a security hole'}")

    # TAMPER 2: corrupt the final conclusion only
    ch3 = build_do_derivation()
    ch3.steps["s3"].recorded = 0.99
    ok3, errs3 = ch3.verify()
    print(f"\n3. faked conclusion (s3=0.99):       verify = {ok3}")
    print(f"   caught: {errs3[0] if errs3 else 'NOTHING'}")

    gate = ok and (not ok2) and (not ok3)
    print("\nQ1-real gate:")
    print(f"  honest chain verifies by self-replay (no oracle) .. {'PASS' if ok else 'FAIL'}")
    print(f"  tampered intermediate is caught ................... {'PASS' if not ok2 else 'FAIL'}")
    print(f"  faked conclusion is caught ........................ {'PASS' if not ok3 else 'FAIL'}")
    print(f"\n  >>> {'PASS — native verification by REAL replay (DP idea made real, no stubs, no energy-poetry)' if gate else 'CHECK'}")
    print("\nWhat this fixes vs DP: every step ACTUALLY re-executes (DP returned the stored output);")
    print("no 'energy conservation' on non-physical steps. What it shares with DP: the replayable-")
    print("chain + tamper-evident hash. This is The One's recompute-credential generalized to a chain.")
    print("Honest scope: tiny hand-built chain; the open part is AUTO-generating such chains during")
    print("native reasoning (so the chain isn't hand-written) — that's the real B5 work ahead.")
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
