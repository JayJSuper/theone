# Auditable Causal Inference — a proposed minimum standard

*The conceptual contribution of The One, separable from any single experiment: a causal-effect estimate intended for a consequential decision should meet four minimum criteria. Each is paired with its failure case — and our LLM-collapse results are the empirical failure case for the first two.*

## The four criteria

### 1. Terminability — provably exact within bounded resources
The computation must be guaranteed to produce the exact answer (or to declare it cannot) within a fixed budget, independent of the question's combinatorial size.
- **Failure case (empirical):** frontier LLMs do not meet this. On exact P(Y|do(X)) with k confounders, accuracy collapses to ≈0 as 2^k grows (deepseek k=4, gpt-5.1 k=5, gemini k=6) — the answer is produced confidently but is wrong, or not produced at all. An explicit enumeration engine meets it by construction.
- **Why it matters:** a risk number that is "usually right but silently wrong when the problem is large" is unusable for decisions whose stakes rise with problem size.

### 2. Independent recomputability — verifiable by a second implementation
The result must be reproducible to tolerance by an independent code path / oracle, not just self-consistent.
- **Failure case (architectural):** LLM outputs are not reproducible — token sampling, context, and model version all move the number; the reasoning chain is not a traceable derivation. Our engine values are recomputed by pgmpy (an independent library) to <1e-6 across 1,207 SCMs.
- **Why it matters:** "trust the output" is not verification. Recomputability turns a claim into a checkable fact, and is what model-risk validation (e.g. SR 11-7) demands of any model output.

### 3. Explicit justification — the credential, not just the number
The result must carry the basis on which it was computed: the adjustment set, the back-door (or front-door / IV) path, the variables conditioned on — auditable by a third party who did not run the computation.
- **Gap in current tools:** standard libraries (pgmpy, DoWhy) return the number, not a structured, auditable justification of *why* that number. An auditor cannot ask "why this adjustment set and not another?" of a bare float.
- **Why it matters:** an auditor / regulator / ethics board needs to interrogate the *reasoning*, not just check the arithmetic. This is the analogue, for causal inference, of proof-carrying code.

### 4. Declared regime & boundary — certifies computation, not the world
The result must state what it assumes and what it does **not** certify: that the causal structure is taken as given, and the CPT's calibration regime (e.g. "normal-times" vs "stressed").
- **Failure case (our own honest limit):** a credential certifies that the computation is exact on the given structure — it does not certify that the structure is correct, nor that the CPT reflects the regime that matters. We quantify this: a normal-times-calibrated credit model is fine on average (bias −0.05) but underestimates **stressed** default by 0.23–0.28 (the 2008 Gaussian-copula failure mode), with a fully valid computation credential.
- **Why it matters:** the most dangerous error is a confidently-exact answer to the wrong question. The boundary must travel with the claim, not hide in an appendix.
- **Extension — from *declared* to *checked* (constraint credentials):** the boundary can be made not only declared but *automatically verified*. A constraint credential attaches deterministic, third-party-recomputable admissibility checks — probability bounds ∈[0,1], a declared monotonicity/effect-sign, normalisation/conservation — to every output. This is a *verifiable* analogue of the energy-based "reject the physically impossible" idea: it catches both an LLM's impossible value (e.g. P=1.0179 emitted unflagged at the cliff) and a misspecified structure's exact-but-sign-violating output (which a bare computation-exact credential would pass), via inequalities anyone can re-check. It remains a *minimum* standard: passing the checks does not make a claim true (the structure can still be wrong); violating one makes it *certainly* inadmissible. (`experiments/constraint_credential/`)

## How the criteria relate to the failure modes
| Criterion | What enforces it here | What fails it |
|---|---|---|
| 1 Terminability | exact enumeration engine | LLM combinatorial collapse |
| 2 Recomputability | pgmpy independent oracle (IPRG) | LLM non-reproducible sampling |
| 3 Justification | structured causal credential | bare numeric output |
| 4 Declared boundary | regime/structure field in credential | silent miscalibration |

## Mapping to model-risk / regulatory requirements
Criteria 2–4 align directly with model-risk management and disclosure regimes — SR 11-7 (independent validation, reproducibility, documented limitations), BCBS 239 (traceability of risk numbers), IFRS 9 / CECL (explainable loss methodology), ECB TRIM (replicable internal models). The standard is not finance-specific, but finance is where its absence is already a named, regulated risk.

## Honest scope of the proposal
This is a **minimum** standard, not a sufficient one. Meeting all four does not make a causal claim *true* — criterion 4 is precisely the admission that structure and calibration can still be wrong. The proposal is narrower and more defensible: *consequential causal-effect estimates should not be accepted unless they at least terminate exactly, recompute independently, justify their adjustment, and declare their regime.* Today, neither LLM outputs nor bare library calls meet all four; an explicit engine with a credential layer can.
