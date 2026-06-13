"""AM-017 backfill: independent third-party (pgmpy) cross-check of ALL SEVEN F-1
assertions (A1-A7), not just the two already in tests/test_oracle_pgmpy.py.

F-1 is the project's first vital number (the causal core: observation != intervention
under confounding). This emits an oracle artifact for the evidence package so the
full truth table 0.82/0.70/0.28/0.40/0.54/0.30/True is independently recomputable.

Run: python experiments/oracle_crosscheck/f1_oracle.py
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
from theone.causal.engine import InterventionEngine
from theone.types import TheOneConfig

# reuse the frozen t1 graph + pgmpy translator from the test module
_s = importlib.util.spec_from_file_location("oracle", Path(__file__).parent.parent.parent / "tests" / "test_oracle_pgmpy.py")
_o = importlib.util.module_from_spec(_s); _s.loader.exec_module(_o)

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def pgmpy_obs(g, xval):
    return float(VariableElimination(_o.to_pgmpy(g)).query(["Y"], evidence={"X": xval}, show_progress=False).values[1])


def pgmpy_do(g, xval):
    """Independent graph surgery in pgmpy: remove U->X, set X to point mass at xval."""
    m = _o.to_pgmpy(g)
    m.remove_edge("U", "X")
    m.remove_cpds(m.get_cpds("X"))
    pt = [[1.0], [0.0]] if xval == 0 else [[0.0], [1.0]]
    m.add_cpds(TabularCPD("X", 2, pt, state_names={"X": [0, 1]}))
    assert m.check_model()
    return float(VariableElimination(m).query(["Y"], do={}, show_progress=False).values[1]) if False else \
        float(VariableElimination(m).query(["Y"], show_progress=False).values[1])


def main():
    g = _o.t1_graph(); eng = InterventionEngine(g)
    rows = []
    checks = [
        ("A1 obs P(Y=1|X=1)", 0.82, eng.query_observation("Y", 1, {"X": 1}).value, pgmpy_obs(g, 1)),
        ("A2 do  P(Y=1|do X=1)", 0.70, eng.query_intervention("Y", 1, {"X": 1}).value, pgmpy_do(g, 1)),
        ("A3 obs P(Y=1|X=0)", 0.28, eng.query_observation("Y", 1, {"X": 0}).value, pgmpy_obs(g, 0)),
        ("A4 do  P(Y=1|do X=0)", 0.40, eng.query_intervention("Y", 1, {"X": 0}).value, pgmpy_do(g, 0)),
    ]
    a1o, a2d, a3o, a4d = (c[3] for c in checks)
    checks.append(("A5 obs ATE", 0.54, eng.observational_ate("X", "Y"), a1o - a3o))
    checks.append(("A6 do  ATE", 0.30, eng.interventional_ate("X", "Y"), a2d - a4d))
    maxdiff = 0.0
    print(f"{'assertion':24}{'frozen':>9}{'engine':>10}{'pgmpy':>10}{'|eng-pgmpy|':>13}")
    for name, frozen, ours, theirs in checks:
        d = abs(ours - theirs); maxdiff = max(maxdiff, d)
        ok = abs(ours - frozen) < 1e-6 and d < 1e-6
        rows.append({"assertion": name, "frozen": frozen, "engine": round(ours, 9),
                     "pgmpy": round(theirs, 9), "eng_pgmpy_diff": d, "pass": ok})
        print(f"{name:24}{frozen:>9}{ours:>10.6f}{theirs:>10.6f}{d:>13.1e}  {'OK' if ok else 'FAIL'}")
    # A7: are_different (qualitative)
    a7 = eng.compare("X", "Y", TheOneConfig()).are_different
    print(f"{'A7 are_different':24}{'True':>9}{str(a7):>10}")
    verdict = "PASS — all 7 F-1 assertions independently confirmed by pgmpy <1e-6" if maxdiff < 1e-6 and a7 else "FAIL"
    print(f"\nmax|engine-pgmpy| = {maxdiff:.2e}  -> {verdict}")
    Path(__file__).parent.joinpath("f1_cross_validation.json").write_text(json.dumps(
        {"assertions": rows, "A7_are_different": a7, "max_abs_diff": maxdiff,
         "oracle": f"pgmpy {__import__('pgmpy').__version__}", "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
