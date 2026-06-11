# The One / 太一

**An open cognitive layer for language models** — explicit causal reasoning (do-operator), persistent memory with provenance, metacognition, and machine-verifiable credentials. 100% open source (Apache-2.0). Belongs to everyone.

## What v0.1.0 actually does (honest scope)

- **Exact causal inference** on small discrete DAGs: observational queries condition properly along backdoor paths (posterior weighting of confounders); interventional queries perform graph surgery (prior weighting). This distinction — the heart of Pearl's do-calculus — is locked by a frozen seven-assertion regression test (`tests/test_confounding_regression.py`, the F-1 fix) plus an A8 mechanism guard against hard-coding.
- **Backdoor identification**: backdoor paths, d-separation blocking (with collider logic), minimal adjustment sets; "not identifiable" is a first-class answer (returns `None`).
- **Synthetic SCM benchmark pipeline** (MVP-2A plumbing): linear-Gaussian SCM generator with confounding strength = standardized coefficient product (frozen Q-C5); two-phase runner with burn-after-use calibration isolation (a frozen-phase run on burned instances hard-fails); EG + RMSE/MAE joint metrics with a frozen conjunctive verdict (Amendments 1/1a); A7 statistical judgment (BCa bootstrap CI ∧ substantive threshold δ_min) that **refuses to run uncalibrated**.
- **Minimal memory store** (SQLite): provenance is mandatory, deletion is real, export = take your data with you.
- **Minimal agent orchestrator v0**: rule-based routing (regex), real computed credentials (recomputable graph hash, adjustment set, timestamp). *No LLM is attached in v0 — see the MOCK-SCOPE note in `src/theone/agent/orchestrator.py`.*

**What it does NOT do yet**: no LLM integration, no counterfactual (level-3) queries, no soft causal graphs, no performance claims of any kind. Every capability statement above is backed by a passing test; nothing here exceeds the test suite.

## Quickstart

```bash
pip install -e ".[dev]"
theone demo causal                      # frozen truth: P(Y=1|X=1)=0.82, P(Y=1|do(X=1))=0.70
theone test                             # full suite
theone bench mvp2a --phase calibrate    # toy-grid calibration (pipeline check only)
```

## Governance

- Charter & blueprint: `docs/charter/` · Frozen-asset registry: `docs/00_FROZEN_REGISTRY.md` · Contribution rules incl. context-hygiene clause: `CONTRIBUTING.md`
- Evidence-first discipline: pre-registration, frozen criteria, burned calibration sets, machine-verified truth tables (R3 v3), claims never ahead of logs. Failures are published, not hidden.
- Mission: return judgment, memory sovereignty, and the standard of honesty to people. **The One is for everyone, by everyone, forever.**
