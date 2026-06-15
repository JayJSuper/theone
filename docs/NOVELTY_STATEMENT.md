# One-page novelty statement — for external assessment

*Purpose: let a domain researcher judge, in ~5 minutes, whether the core finding is new — and tell us where it isn't. We would rather learn "X already did this" now than after submission. Please push toward "known."*

## The finding (one paragraph)
We benchmark frontier LLMs (gpt-5.1, gemini-2.5-pro, deepseek-v4, claude) on **exact** interventional queries P(Y=1 | do(X=1)) in discrete confounded SCMs, sweeping k = number of independent confounders that must be marginalized (2^k configurations). Models solve textbook-size cases (k=1–2: ≈100%) — so this is **not** a knowledge gap — but exact accuracy (within ±0.005) collapses as 2^k grows, and the collapse is **universal and capability-ordered**: the cliff's location slides monotonically with model capability — deepseek-v4-flash at 2⁴=16, gpt-5.1 at 2⁵=32, gemini-2.5-pro at 2⁶=64, and the frontier claude-opus-4.8 holds to 2⁶ (0.85) then collapses at 2⁷=128 (0.33→0.08, a computational collapse with no protocol failures). No tested model escapes the cliff; spelling the correct back-door recipe into the prompt does not rescue any of them. An explicit enumeration engine is exact (1.000) at every k, at ~zero token cost, with every value independently recomputed by pgmpy to <1e-6.

## What is standard (NOT claimed as new)
- The inference itself: exact marginalization / variable elimination over discrete CPTs, and back-door adjustment. Textbook; available in pgmpy, DoWhy. We use pgmpy as an independent verifier, not a contribution.
- The general phenomenon "LLM reasoning degrades with compositional/multi-step complexity" (compositional gap, multi-hop, etc.). Well documented. **We do not claim this is new.**

## What we believe may be new (the precise question for you)
That the LLM failure on causal queries is specifically **exact-marginalization terminability** — cleanly indexed by the single axis 2^k, consistent across four model families, with a capability-dependent but never-eliminated collapse point, and unfixable by prompting — and that the natural remedy is to move the marginalization onto a deterministic engine and attach a **third-party-recomputable causal credential** (adjustment set, back-door path, calibration regime). We have not found this specific localization in the LLM-causal-benchmark literature (e.g. CLadder, Corr2Cause, causal-probing work), which tests whether LLMs answer causal questions rather than placing LLMs and exact inference on a common combinatorial-load axis. **Is this localization already characterized? If so, where?**

## Two secondary findings (also: new or known?)
1. **Prompt engineering cannot fix it.** A causal-inference scaffold prompt does not help and can hurt — it degrades a struggling model, does nothing for a comfortable one, and cannot rescue a collapsed one (0.00 either way).
2. **A benchmarking artifact we had to correct.** At high k, the true do-probability concentrates near the mean of the visible CPT cells, so a collapsed model that "anchors on the typical visible value" looks deceptively accurate (MAE understated; spurious non-monotonic recovery). De-anchoring the generator removes it and the collapse becomes monotone. We flag this as a control any probabilistic-reasoning benchmark should apply. (Known practice?)

## Reproducible anchors
De-anchored generator, 4-decimal precision, protocol-failure = error. Accuracy(±0.005) cliff: deepseek k=4, gpt-5.1 k=5, gemini-2.5-pro k=6, claude-opus-4.8 k=7 (capability-ordered); engine 1.000 throughout; 1,207 SCM truths pgmpy-verified <1e-6. Code: `github.com/JayJSuper/theone` (`experiments/complexity_axis`, `experiments/deanchor_cliff`).

## What would settle it
One sentence from you: *"this specific characterization is new"* — or *"X (year) already did it,"* with a pointer. Either answer is valuable; the second is more valuable.
