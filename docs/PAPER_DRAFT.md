# When Causal Correctness Becomes Verifiable: An Explicit Intervention Engine vs. Frontier LLMs under Combinatorial Load

*The One (太一) project — systematic paper, draft v1. All numerical results are committed and reproducible; artifact paths are given inline. `[TODO]` marks figures and related-work citations to be added.*

---

## Abstract

Frontier large language models (LLMs) know textbook causal inference: given a small confounded system in textbook form, GPT-5.1, DeepSeek-v4, Gemini-2.5-Pro and Claude all compute the back-door adjustment correctly. We show that correct causal-effect estimation is nonetheless not primarily a *knowledge* problem but a *computational-terminability* problem. As the number of confounder configurations that an intervention query must marginalize, 2^k, grows, every LLM we tested fails — not by lacking the method, but by being unable to carry out the exact marginalization within its reasoning budget. An explicit intervention engine that performs the marginalization deterministically is exact (accuracy 1.000) across the entire range, in milliseconds, at essentially zero token cost, and attaches to every answer a credential that an independent third-party library (pgmpy) recomputes to within 10⁻⁶. We establish this with three vital numbers under a known causal graph and given conditional-probability tables (CPTs): a confounding-core regression test, a complexity-scaling axis, and a combinatorial-explosion axis. We then relax the toy premises one step at a time — CPTs estimated from finite data, an imperfect adjustment set, and adversarial true-value distributions — and find the engine's advantage survives, while honestly delimiting where it does *not*: at low combinatorial complexity in clean natural language the LLM needs no engine, and a credential certifies computation, **not** structural correctness. The combinatorial cliff is universal across model families, but its *location* slides with model capability and token budget; it is never eliminated. We argue the contribution is making "which causal output can I trust" a machine-verifiable fact rather than a gamble, within a clearly bounded regime.

> **Scope of every "1.000" and "≈850×" in this paper**: results hold under a *known* causal graph with *given* CPTs (structured input). For natural-language and unknown-structure settings, see §6 (Boundaries).

---

## 1. Introduction

Causal questions — *what happens to Y if we intervene on X?* — are distinct from predictive ones, and the distinction is the back-door adjustment: under confounding, P(Y|do(X)) ≠ P(Y|X). A decade of work (Pearl, Spirtes, Shpitser) makes the *math* standard. The open question for the LLM era is operational: when a model is asked an interventional query, does it reliably *carry out* the computation?

We separate two failure modes. A **knowledge** failure is not knowing that confounding must be adjusted. A **terminability** failure is knowing the method but being unable to execute it within a fixed reasoning budget. We find frontier models have essentially no knowledge failure on textbook-form problems, and a severe, systematic terminability failure as combinatorial load grows. This reframes the value of an explicit causal layer: not to teach the model causal inference, but to move the combinatorial computation off the autoregressive reasoning chain and onto a deterministic engine that is exact and that emits a verifiable credential.

Our contributions:
1. Three reproducible "vital numbers" isolating the terminability failure (§3), each independently recomputed by a third-party library.
2. An ecological-validity gradient showing the engine's edge survives finite-sample CPTs and imperfect structure, and a methodological artifact (anchoring) that benchmarks of probabilistic reasoning must control for (§4–5).
3. An explicit, honest delimitation of where the engine does not help, including the limit of what a credential certifies (§6).

---

## 2. Setup and methods

**Systems.** Binary structural causal models (SCMs) with a treatment X, outcome Y, k independent confounders U₁…U_k each pointing to both X and Y (pure back-door), plus distractor nodes. The adjustment set is {U₁…U_k}; the interventional target is P(Y=1|do(X=1)) = Σ_u P(Y=1|X=1,u)∏_i P(u_i).

**Subjects.** (A) raw LLM, (B) LLM with a causal-inference scaffold prompt, (C) the explicit engine (exact enumeration / graph surgery). LLMs: gpt-5.1, deepseek-v4-flash, gemini-2.5-pro, claude. Each is given the full SCM in text and asked for a 4-decimal probability.

**Scoring (AM-007).** A protocol failure — no parseable answer within a fixed budget — counts as an error for accuracy (terminability is a capability dimension), is *excluded* (not substituted) from continuous-error means, and its rate is reported separately (AM-007+).

**Verification (the immune system).**
- *R3*: every frozen truth is hand-derived and machine-recomputed; the machine value is authoritative.
- *Implementation gate (IPRG, AM-016)*: the engine's value is recomputed by an independent code path (pgmpy variable elimination + graph surgery); agreement to 10⁻⁶ certifies the absence of implementation bugs.
- *Semantic gate (AM-016+)*: a statement⟺structure⟺output consistency check certifies the absence of specification errors (e.g. a problem text inconsistent with the SCM used as truth). Two gates, each with a defined blind spot, neither sufficient alone.
- *Beautiful Failures*: an append-only log of falsified hypotheses and retractions, including one retraction made *after* a result was frozen (§6).

---

## 3. Layer 1 — Three vital numbers (known structure, given CPTs)

The engine is exact (1.000) on all three. Across the three, **1,207 SCM truths were independently recomputed by pgmpy**, max |pgmpy − engine| < 10⁻⁶ (AM-017).

### 3.1 The causal core (F-1)
A single confounded SCM with the frozen truth table P(Y=1|X=1)=0.82 vs P(Y=1|do(X=1))=0.70 (and four further assertions). All seven assertions are reproduced by pgmpy to a difference of **0.0** (`experiments/oracle_crosscheck/f1_oracle.py`). This is a sanity anchor: the engine does the right thing exactly where naive conditioning is biased by 0.12.

### 3.2 Complexity scaling
Three node-count tiers (5/8/12) × 150 instances × {A,B,C}. The flash-tier LLM degrades with causal complexity (large-tier accuracy 0.45–0.61) while the engine stays at 1.000; the separation has p = 3.5×10⁻¹⁸. A fixed-skeleton control that adds only *irrelevant* nodes does not degrade any system, isolating the cause as causal complexity, not node count. All 900 truths are pgmpy-verified and match the frozen formal-run values exactly (`scale_oracle.py`).

### 3.3 The combinatorial cliff (headline)
Sweeping k with the x-axis as 2^k, accuracy on the interventional query collapses for every LLM while the engine is flat at 1.000 (millisecond latency). We report the cliff on the *de-anchored* generator (the cleanest; see §5) as the main result: gpt-5.1 accuracy falls **monotonically** 0.64 (k=4) → 0.04 (k=5) → 0.00 (k=6), with mean absolute error rising to 0.24 at k=5; the engine is 1.000 throughout (IPRG pass, 4.98×10⁻⁷). Cross-family on the same generator, deepseek/gpt-5.1/gemini-2.5-pro all reach accuracy 0.00 by k=6. `[TODO: Figure 1]`

---

## 4. Layer 2 — Ecological validity

We relax the toy premises one at a time. Each step keeps the engine and the LLM on equal footing (same information) and asks whether the engine's edge survives.

### 4.1 CPTs estimated from finite data
Instead of exact CPTs, both engine and LLM receive the *same* table MLE-estimated from n ∈ {50,200,1000} samples (a fairness constraint: the LLM never sees raw data, which would conflate discovery with inference). The engine's error is exactly the irreducible *estimation floor* (the best any method can do with that data); it shrinks with n (0.076→0.018 at k=1) and, notably, does **not** explode with k, because per-cell estimation errors partially cancel under marginalization. The LLM adds reasoning error on top, so the engine's advantage **survives and emerges at k≥4**; at low k the two tie. This is the first end-to-end pass through the IPRG gate (360 instances, 4.99×10⁻⁷). `experiments/cpt_finite_sample/`

### 4.2 An imperfect adjustment set
If a confounder is *omitted* from the adjustment set, the engine still computes exactly — on the wrong structure. Omitting a weak confounder degrades gracefully (bias 0.009–0.043, still beating the LLM's high-k collapse); omitting a **strong** confounder yields a silent 0.30 bias delivered with a valid *computation* credential. This delimits the credential precisely: it certifies that the computation is exact on the given structure, **not** that the structure is correct (§6). `experiments/wrong_structure/`

---

## 5. Layer 3 — Supplementary findings

### 5.1 Prompt engineering cannot fix the terminability failure
We tested a minimal prompt (A) vs a full causal-inference scaffold (B) across three regimes, and the scaffold helps in none of them:
- **Struggling model** (deepseek-v4-flash, 12-node tier): the scaffold *hurts* — accuracy 0.447 (B) vs 0.613 (A), with higher latency and more protocol failures.
- **Comfortable model** (gpt-5.1, 12-node tier): no effect — A 0.967 vs B 0.933 (n=30), within noise; the model already handles the tier.
- **Collapsed model** (gpt-5.1, de-anchored k=5, deep in the cliff): the scaffold cannot *rescue* it — both A and B reach **0.00 accuracy** (MAE 0.145 vs 0.162, n=25); once 2^k exceeds the budget, no system prompt recovers it.

The unifying finding is stronger than "scaffolds are harmful": **prompt engineering — including the exact, correct back-door recipe spelled out in the prompt — does not solve the combinatorial cliff.** The failure is computational, not instructional; it requires moving the computation off the reasoning chain. `experiments/scaffold_crossbase/`

### 5.2 Two crash signatures
LLM failure has two distinct forms, which are the twin targets of the credential philosophy. gpt-5.1 fails by **confident wrong answers**: at k≥5 it returned 87/98 precise-to-4-decimals, unwarned, wrong probabilities (mean error 0.047, max 0.45), including mathematically impossible values P>1 (e.g. 1.0179, 1.28). deepseek and claude fail by **protocol collapse**: the reasoning chain exhausts the budget and returns nothing. Both are terminability failures.

### 5.3 The anchoring artifact (a methodology contribution)
At high k, the true interventional probability concentrates near the mean of the visible CPT cells (a central-limit effect). A collapsed LLM that anchors on that visible mean lands near the truth by coincidence — inflating its apparent high-k performance and depressing its MAE. We verify this is an artifact, not capability, with two controls: a skewed-CPT generator (true value moves to 0.71, yet the artifact persists because it relocates with the mean) and a **de-anchored** generator (extreme confounder marginals spread the true value across [0,1] so no single anchor works). On the de-anchored generator the LLM's accuracy collapses monotonically (the spurious "k=6 recovery" vanishes) and MAE rises to 0.24. **Recommendation: benchmark high-k probabilistic reasoning by exact accuracy, not MAE.** `experiments/skewed_cpt_robustness/`, `experiments/deanchor_cliff/`

### 5.4 A financial application: credit-risk causal attribution
The combinatorial advantage applies where causal inference is genuinely *discrete*. Linear-Gaussian factor-return models have closed-form interventional effects and no cliff; **credit default** does not — default and firm distress are binary and multiple systemic stress factors create a real 2^k marginalization. We model k binary systemic factors (rate shock, liquidity freeze, credit-spread widening, …) as common causes of firm distress D and default Y, and ask P(Y=1|do(D=1)) — the causal effect of distress on default under a stress intervention, adjusting for systemic confounders. Because tail events are rare (P(factor)∈[0.03,0.20]), the true value is naturally de-anchored. gpt-5.1 collapses (accuracy 0.32 → 0.04 → 0.00 at k=4,5,6) with mean error ~0.10 on a default probability — a capital/pricing-material error delivered with 4-decimal confidence — while the engine is exact (IPRG 4.89×10⁻⁷). The value proposition here has two legs, and in finance the second may dominate: (i) the cliff (LLMs cannot reliably attribute multi-factor default risk), and (ii) the **credential** — a third-party-recomputable, auditable causal attribution that maps directly onto model-risk-management / regulatory validation (SR 11-7). `experiments/finance_credit_risk/`

### 5.5 Efficiency
The engine costs ~0 tokens and ~0.07s. LLMs spend 63–79% of their tokens on wrong answers; at high k, deepseek spent 917k tokens to get 0/100 correct. Token-normalized accuracy (AM-013) makes the cost asymmetry explicit. (Scope: structured input.)

---

## 6. Layer 4 — Boundaries and open challenges

Honesty travels with the claim.

- **The cliff's location is budget/capability-dependent.** It is universal — every LLM hits it — but a heavier reasoner with a larger budget pushes it back a notch: deepseek fails at k=4, gpt-5.1 at k=5, gemini-2.5-pro (given a 6× output budget to accommodate its thinking tokens) holds k=4 at 1.00 and k=5 at 0.40 but still collapses to 0.00 with protocol failures at k=6. The correct claim is therefore *"every LLM hits the cliff; its location moves back with budget but is never eliminated; the engine is exact at all k for ~0 cost"* — not "LLMs cannot do k≥4."
- **The engine is itself O(2^k).** Exact enumeration also scales exponentially; the engine has its own far-out limit (k≈25–30). Its advantage window is "k large enough to break LLM reasoning, small enough to enumerate," which comfortably covers where LLMs fail (k=4–6) but is not unbounded. The claim is *exact-and-cheap vs approximate-and-expensive*, not *tractable vs intractable*.
- **Known-structure assumption.** All results assume the DAG is given. Whether the high-k-known-structure regime is common in real medical/financial data is a question for domain experts.
- **Low-complexity natural language.** When numbers are correctly stated in prose and k is small, the LLM marginalizes correctly itself and the engine offers no measurable advantage (k=1: 24/24 exact; k=2: MAE 10⁻⁴).
- **Qualitative natural language.** When the SCM is purely qualitative, the numeric comparison is ill-posed (a retracted result, REJ-002 / WEAK-03), and natural-language structure extraction is ~65% reliable.
- **A retracted result, kept public.** An earlier "engine wins on prose" finding was a two-canceling-bug artifact, caught and retracted *after* it had been frozen (REJ-002). It is recorded, not hidden — a test of the methodology.
- **The counterfactual-gradient bet (T4) is unresolved.** A separate line on reverse-engineering generalization shows a synthetic→real out-of-distribution degradation of 12×; it remains an open challenge, stated here rather than omitted.
- **Causal discovery (unknown DAG)** is the last unproven frontier, motivated directly by §4.2: because structural error is silent and unbounded, *how to obtain the structure reliably* is the decisive question for real-world deployment.

---

## 7. Conclusion

Within a known-structure, given-CPT regime that comfortably contains where frontier LLMs break, an explicit intervention engine converts causal correctness from a gamble into a machine-verifiable fact: exact at all combinatorial loads, milliseconds, ~zero tokens, third-party-recomputable. The boundary is equally important: a credential certifies computation, not structure, and the regime is bounded by enumerability and by the assumption of a known graph. The combinatorial cliff is universal across model families and slides with budget but is never eliminated — which is precisely why moving the combinatorial part of causal computation off the reasoning chain, and certifying it, is worth doing.

---

## Reproducibility
100% open source: `github.com/JayJSuper/theone`. Key artifacts: cliff `experiments/complexity_axis/` + `deanchor_cliff/`; ecological validity `experiments/cpt_finite_sample/`, `wrong_structure/`, `skewed_cpt_robustness/`; oracles `experiments/oracle_crosscheck/` + `complexity_axis/cross_validate.py`; registry `docs/00_FROZEN_REGISTRY.md`; failures `docs/BEAUTIFUL_FAILURES.md`.
