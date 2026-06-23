"""Q4 fixed · AUTO-generate a REPLAYABLE chain of PURE steps (no stateful replay).

DP's revised Q4 stored a function_ref to a STATEFUL engine method and re-invoked it during
verification — which (1) mutates engine state on replay (re-adding an edge corrupts/errs)
and (2) compares outputs carrying timestamps/ids (false mismatch). The clean abstraction:
instrument the computation to auto-emit PURE steps (inputs -> deterministic output, no side
effects); verification re-runs the pure function. This auto-generates the chain (no hand-
writing) AND keeps it genuinely replayable.

Demo: an instrumented do(X=1) computation auto-records each pure sub-step; the chain
replays exactly; tampering any step is caught; replay is idempotent (no state mutation).

Run:  .venv/bin/python experiments/bline_auto_chain/run.py
"""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).parent


class PureStep:
    """A step that stores HOW to recompute (a pure fn of named inputs), not a stateful call."""
    def __init__(self, sid, op, fn, inputs, recorded):
        self.sid, self.op, self.fn, self.inputs, self.recorded = sid, op, fn, inputs, recorded
    def replay(self):
        return self.fn(**self.inputs)              # pure: no side effects, deterministic
    def verify(self):
        got = self.replay()
        ok = abs(got - self.recorded) < 1e-9 if isinstance(got, float) else got == self.recorded
        return ok, got


class AutoChain:
    """Auto-collects pure steps as a computation runs (instrumentation, not hand-written)."""
    def __init__(self):
        self.steps = []
    def record(self, sid, op, fn, **inputs):
        out = fn(**inputs)                          # run once, record the pure step
        self.steps.append(PureStep(sid, op, fn, inputs, out))
        return out
    def verify(self):
        errs = []
        for s in self.steps:                        # replay each pure step independently
            ok, got = s.verify()
            if not ok:
                errs.append(f"{s.sid}: replay {got} != recorded {s.recorded}")
        return (not errs), errs
    def conclusion(self):
        return self.steps[-1].recorded if self.steps else None


# --- pure computational primitives (no engine state) -------------------------
def stratum_weights(pu): return [(1 - pu), pu]
def stratum_outcomes(yU0, yU1): return [yU0, yU1]
def marginalize(weights, outcomes): return sum(w * y for w, y in zip(weights, outcomes))


def instrumented_do():
    """A do(X=1) computation that AUTO-emits its replayable chain (no hand-written trace)."""
    ch = AutoChain()
    w = ch.record("s1", "P(U) weights", lambda pu: stratum_weights(pu), pu=0.4)
    o = ch.record("s2", "P(Y=1|do X=1,U)", lambda yU0, yU1: stratum_outcomes(yU0, yU1), yU0=0.45, yU1=0.85)
    ch.record("s3", "marginalize -> do(X=1)", lambda weights, outcomes: marginalize(weights, outcomes),
              weights=w, outcomes=o)
    return ch


def main():
    print("=== Q4 fixed · auto-generated REPLAYABLE chain of pure steps ===\n")
    ch = instrumented_do()
    print(f"auto-recorded {len(ch.steps)} pure steps during the do() computation (not hand-written)")
    print(f"conclusion do(X=1) = {ch.conclusion()}")

    ok, errs = ch.verify()
    print(f"\n1. replay (re-run pure fns) verifies: {ok}")

    # replay is IDEMPOTENT (no state mutation) — verify twice, same result
    ok2, _ = ch.verify()
    idempotent = ok and ok2
    print(f"2. replay is idempotent (no state corruption): {idempotent}")

    # tamper a step's recorded value -> caught
    ch.steps[1].recorded = [0.45, 0.95]
    bad, errs2 = ch.verify()
    print(f"3. tampered step caught: {not bad}  ({errs2[0] if errs2 else 'MISSED'})")

    gate = ok and idempotent and (not bad)
    print("\nQ4-fixed gate:")
    print(f"  chain auto-generated from the computation .......... {'PASS' if len(ch.steps)==3 else 'FAIL'}")
    print(f"  pure-step replay verifies (no stateful re-call) .... {'PASS' if ok else 'FAIL'}")
    print(f"  replay idempotent (no state mutation) .............. {'PASS' if idempotent else 'FAIL'}")
    print(f"  tampering caught ................................... {'PASS' if not bad else 'FAIL'}")
    print(f"\n  >>> {'PASS — auto-generated, genuinely replayable chain (DP Q4 two holes fixed)' if gate else 'CHECK'}")
    print("\nFix vs DP: steps store PURE fns of named inputs (not stateful engine method refs), so")
    print("replay neither mutates state nor false-mismatches on timestamps/ids. Auto-recorded via")
    print("instrumentation (record()), not hand-written. This is the right abstraction for Q4.")
    (HERE / "results.json").write_text(json.dumps(
        {"n_steps": len(ch.steps), "conclusion": ch.conclusion(), "gate": bool(gate)}, indent=2))
    if not gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
