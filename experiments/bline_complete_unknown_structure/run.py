"""③ COMPLETE FORM on UNKNOWN structure — discover (no order) -> identifiable do -> speak / abstain.

NOTE-137/138's integrated system assumed the causal structure for the do step. NOTE-142 gave
order-free discovery with identifiability-aware abstention. ③ ties them: one pipeline that, given
only observational data over named variables (NO order, NO edges), DISCOVERS the structure, decides
whether the target causal query is identifiable, and either SPEAKS a fluent engine-audited answer or
ABSTAINS — never committing a non-identifiable effect. The whole system now runs without assuming
structure.

Pipeline:  data -> order-free discover (BIC confidence set) -> IDA invariance test
           -> identifiable? VerifiedReporter renders fluent + round-trip-gated + engine-audited
           -> else "inconclusive".

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_complete_unknown_structure/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("ofd", ROOT / "experiments/bline_order_free_discovery/run.py")
ofd = _ilu.module_from_spec(_spec); sys.modules["ofd"] = ofd; _spec.loader.exec_module(ofd)
from theone.language import VerifiedReporter, Finding
FAST = os.environ.get("THEONE_FAST") == "1"


def discover_and_query(S, X, Y):
    """order-free discovery + IDA: returns (identifiable, ate_estimate, true_in_class_ok, klass_size)."""
    DAGS = ofd.all_dags()
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def nscore(i, parents): return ofd.fit_node_score(S, i, parents)
    scored = [(sum(nscore(i, tuple(ofd.parents_of(e, i))) for i in range(ofd.K)), e) for e in DAGS]
    best = max(s for s, _ in scored)
    klass = [e for s, e in scored if best - s < 7.0]
    effects = []
    for e in klass:
        cb = np.zeros(ofd.K); cw = np.zeros((ofd.K, ofd.K))
        for i in range(ofd.K):
            wf, b = ofd.fit_cpt(S, i, tuple(ofd.parents_of(e, i))); cw[:, i] = wf; cb[i] = b
        effects.append(ofd.do_effect(e, cb, cw, X, Y))
    spread = max(effects) - min(effects)
    return spread < 0.05, float(np.mean(effects)), klass


def main():
    print("=== ③ COMPLETE FORM on UNKNOWN structure (discover -> identifiable do -> speak/abstain) ===\n")
    reporter = VerifiedReporter(
        label={"x": "the intervention", "y": "the outcome"},
        entity_syn={"x": ["the intervention", "intervening", "the treatment", "acting on x"],
                    "y": ["the outcome", "y", "the result", "the target"]})
    rng = np.random.default_rng(1)
    n_obs = 6000 if FAST else 12000

    # a few illustrative runs (show spoken output)
    print("  illustrative runs:")
    shown_spk = shown_abs = 0
    for _ in range(40):
        edges, bias, w = ofd.random_dag(rng)
        X = int(rng.integers(0, ofd.K)); Y = int(rng.integers(0, ofd.K))
        while Y == X: Y = int(rng.integers(0, ofd.K))
        true_ate = ofd.do_effect(edges, bias, w, X, Y)
        S = ofd.sample_data(edges, bias, w, n_obs, rng)
        ident, ate, klass = discover_and_query(S, X, Y)
        if ident and shown_spk < 2:
            if abs(ate) < 0.05:                               # identifiably NULL — do not fake a direction
                line = f"No detectable causal effect of the intervention on the outcome (identifiable as ~zero, ATE={ate:+.2f})."
            else:
                f = Finding(cause="x", effect="y", direction=1 if ate >= 0 else -1, zone="VERIFIABLE", ate=ate, e_value=2.0)
                out = reporter.report([f]); line = (out["report"] or out["held_back"])[0]
            print(f"    [identifiable, true {true_ate:+.2f}] SPEAK → \"{line}\""); shown_spk += 1
        elif not ident and shown_abs < 2:
            print(f"    [not identifiable: {len(klass)} DAGs disagree on the effect] ABSTAIN → \"inconclusive\""); shown_abs += 1
        if shown_spk >= 2 and shown_abs >= 2: break

    # system-level red-line over many scenarios
    print("\n  system-level red-line (unknown structure):")
    N = 50 if FAST else 100
    spoke = abst = wrong = correct = 0; errs = []
    rng2 = np.random.default_rng(7)
    for _ in range(N):
        edges, bias, w = ofd.random_dag(rng2)
        X = int(rng2.integers(0, ofd.K)); Y = int(rng2.integers(0, ofd.K))
        while Y == X: Y = int(rng2.integers(0, ofd.K))
        true_ate = ofd.do_effect(edges, bias, w, X, Y)
        S = ofd.sample_data(edges, bias, w, n_obs, rng2)
        ident, ate, _ = discover_and_query(S, X, Y)
        if ident:
            spoke += 1; errs.append(abs(ate - true_ate))
            if abs(true_ate) > 0.1:
                if np.sign(ate) == np.sign(true_ate): correct += 1
                else: wrong += 1
        else:
            abst += 1
    print(f"    scenarios={N}: SPOKE={spoke} (err {np.mean(errs) if errs else 0:.3f}), ABSTAINED={abst}, "
          f"clear-effect correct={correct}, CONFIDENTLY-WRONG={wrong}")

    g1 = wrong == 0
    g2 = (np.mean(errs) if errs else 1) < 0.06
    g3 = spoke >= 1 and abst >= 1
    allok = g1 and g2 and g3
    print("\nunknown-structure complete-form gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] RED-LINE: 0 confidently-wrong spoken answers (unknown structure)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] identifiable spoken effects accurate (<0.06)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] system both speaks (identifiable) and abstains (non-identifiable)")
    msg = ("PASS — the complete form runs WITHOUT assuming structure: discover (no order) -> identifiable "
           "do -> fluent engine-audited speech, abstaining on non-identifiable queries. The full system "
           "is honest end-to-end even when the causal graph is unknown.") if allok else "CHECK"
    print(f"\n  >>> {msg}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
