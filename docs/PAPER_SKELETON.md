# System Paper — Skeleton (working draft scaffold)

> Backbone for the systematic paper. Four-layer narrative (gatekeeper-endorsed): core → ecological validity → supplementary → boundaries. Every number here is committed + reproducible; artifact paths given. `[TODO]` marks unwritten prose or undecided placement. **Numbers are frozen facts; the prose is to be written.**

**Working title**: *When Causal Correctness Becomes Verifiable: An Explicit Intervention Engine vs. Frontier LLMs under Combinatorial Load*

**One-line thesis**: Correct causal reasoning is not a knowledge problem (frontier LLMs know the textbook math) but a **computational-terminability** problem; an explicit causal engine with a machine-verifiable credential is correct and exact where LLMs collapse — within an honestly-bounded regime (known structure).

---

## Abstract `[TODO write]`
Must contain, at first mention of "1.000": the scope clause — *"under a known causal graph and given conditional-probability tables; for natural-language / unknown-structure settings see §Boundaries."* Anti-misquote shield (per Jack). Efficiency claim ("≈850×, ~0 tokens") likewise scope-qualified to structured input.

---

## Layer 1 — Core: three vital numbers
The engine is exact (1.000) on all three; **every truth independently recomputed by pgmpy** (1,207 SCMs total, max |pgmpy−engine| < 1e-6 — AM-017).

### 1.1 F-1 — the causal core (observation ≠ intervention)
- Frozen truth table (R3): 0.82 / 0.70 / 0.28 / 0.40 / 0.54 / 0.30 / True (A1–A7).
- Independent: all 7 confirmed by pgmpy, diff **0.0** (`experiments/oracle_crosscheck/f1_oracle.py`).
- Role: **sanity anchor**, not headline (per Jack). Shows the engine does the right thing where naive conditioning fails.

### 1.2 Scale axis — complexity, not node count
- 3 tiers (5/8/12 nodes) × 150 × {raw LLM A, LLM+scaffold B, engine C}.
- Separation **p = 3.5×10⁻¹⁸**; flash-tier LLM degrades (L: 0.45–0.61), engine 1.000.
- Clean-skeleton control (fixed structure + distractor nodes): all three saturate → **collapse is driven by causal complexity, not node count**. (Control, embedded, not its own section.)
- Independent: **900/900** pgmpy-verified, matches frozen truths exactly (`scale_oracle.py`).

### 1.3 k-axis — the combinatorial cliff (HEADLINE)
- k = adjustment-set cardinality; x-axis 2ᵏ (configs to marginalize). Pure back-door.
- **Main text uses the DE-ANCHORED generator** (NOTE-006, cleanest — see §3.3 why): gpt-5.1 accuracy collapses **monotonically** k4 0.64 → k5 0.04 → **k6 0.00**; MAE explodes to **0.24** at k5; engine 1.000 throughout (IPRG PASS).
- Cross-family (uniform generator, AM-011): 4 bases × 3 families (gpt-5.1, deepseek, gemini, claude) all collapse at 2⁵=32. `[TODO: fold de-anchored gemin​i cross-family result when gemini_check completes]`
- Two crash signatures (→ §3.2).

---

## Layer 2 — Ecological validity (the engine edge past toy conditions)
Gradient relaxing the "exact CPT + perfect structure" premise, one step at a time.

### 2.1 Estimated CPT (AM-018)
- CPT MLE-estimated from n∈{50,200,1000}; **engine and LLM get the SAME estimated table** (fairness gate).
- Engine pays only the irreducible estimation floor (shrinks with n; does NOT explode with k — cell errors cancel under marginalization). LLM adds reasoning error on top → **engine advantage survives, emerges at k≥4**. Low-k tie.
- IPRG first end-to-end run (360 instances, 4.99e-7). `experiments/cpt_finite_sample/`

### 2.2 Imperfect structure (NOTE-004)
- Missing a weak confounder → graceful degradation (bias 0.009–0.043; engine still beats LLM collapse at high k).
- Missing a **strong** confounder → **silent 0.30 bias** with a valid computation credential.
- **The credential certifies computation, NOT structure** — honest limit; motivates the causal-discovery frontier. `experiments/wrong_structure/`

### 2.3 True-value-distribution robustness (NOTE-005, NOTE-006)
- Skewed CPT (Beta) and de-anchored (extreme marginals) generators. Core cliff robust; removing the anchoring artifact makes the LLM look **worse** and the engine advantage **larger** (→ §3.3). `experiments/skewed_cpt_robustness/`, `experiments/deanchor_cliff/`

---

## Layer 3 — Supplementary findings
### 3.1 Scaffolding is harmful at scale
- Raw LLM (A) **>** LLM+scaffold (B) under complexity: B doubles latency, more protocol failures, lower accuracy (L: 0.447 vs 0.613). Challenges "add causal hints to help." `[TODO: confirm cross-base beyond deepseek; currently solid on deepseek]`

### 3.2 Two crash signatures (credential philosophy's twin targets)
- gpt-5.1 → **confident wrong** (k≥5: 87/98 wrong-with-no-warning, incl. impossible P>1, e.g. 1.0179 / 1.28). `CONFIDENT_WRONG.md`
- deepseek/claude → **protocol failure** (reasoning exhausts budget; AM-007). Both are terminability failures, not knowledge failures.

### 3.3 The anchoring artifact (a methodology contribution)
- At high k, true_do ≈ mean of visible CPT cells (CLT). A collapsed LLM anchoring on that mean gets lucky → understated apparent failure (the "k6 recovery" and small MAE were artifacts).
- **Fix**: de-anchored generator (extreme marginals) spreads true_do off the visible mean. Result: monotone collapse to 0.00, MAE 0.24. **Use exact accuracy (±0.005), not MAE, at high k.** This is a transferable caveat for anyone benchmarking probabilistic reasoning.

### 3.4 Efficiency / cost
- Engine ~0 tokens, 0.07s; LLMs burn 63–79% of tokens on wrong answers; deepseek 917k tokens → 0/100 correct at high k. Token-normalized accuracy (AM-013). Scope: structured input.

---

## Layer 4 — Boundaries & open challenges (honesty travels with claims)
- **Known-structure assumption** (AM-008): all core results assume the DAG is given. The headline regime where the engine wins (high k, known structure) — ecological reality TBD by domain experts `[expert Q]`.
- **Low-k natural language**: when numbers are correctly stated in prose and k small, the LLM marginalizes correctly itself — engine no advantage (wildtext numeric). `[main text or discussion]`
- **Qualitative natural language**: numeric comparison ill-posed (WEAK-03); structure-extraction ~65% reliable.
- **Bet ② (counterfactual gradient, T4)**: WEAK — synthetic→real OOD degradation 12×. State openly in Discussion (per Jack), do not hide: *"reverse-engineering generalization faces significant obstacles; ongoing."*
- **Causal discovery (unknown DAG)**: last unproven frontier; `docs/NEXT_FRONTIER_causal_discovery.md`. Motivated by §2.2 (silent structural bias).
- **A retracted result, kept public**: REJ-002 (wildtext double-bug artifact, retracted after freeze). Demonstrates the discipline.

---

## Methodology section (the immune system)
- **R3** (hand-calc + machine-recompute, machine authoritative) for frozen truths.
- **IPRG / implementation gate** (AM-016): engine vs independent path (pgmpy) — catches implementation bugs.
- **Three-way / semantic gate** (AM-016+): statement⟺structure⟺output — catches REJ-002-class spec errors. Two gates, each with a defined blind spot, neither sufficient alone.
- **Protocol-failure handling** (AM-007 / AM-007+): = error for accuracy; excluded (not substituted) from continuous-error means; failure-rate reported separately.
- **《Beautiful Failures》** (append-only): BF-01/02, SAT-01/02, WEAK-01/02/03, REJ-001/002 + bidirectional self-retractions. Negative results published alongside positive.

---

## Figures `[TODO]`
- F1: the de-anchored cliff (accuracy vs 2ᵏ, engine flat 1.000) — main figure.
- F2: three-generator robustness (uniform/skewed/de-anchor) — appendix.
- F3: ecological-validity gradient (estimated CPT, imperfect structure).
- F4: cost (tokens-to-wrong-answer).

## Status of inputs
- Ready: §1.1–1.3, §2.1–2.3, §3.2–3.4, §4, methodology — all data committed.
- Pending: §1.3 de-anchored gemini cross-family (running); §3.1 scaffold cross-base; wildtext placement decision; bet② framing.
