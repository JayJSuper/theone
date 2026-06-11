# Contributing

## Frozen assets
Anything listed in `docs/00_FROZEN_REGISTRY.md` may only be changed via an explicit
amendment that references the asset ID. Status declarations ("approved / frozen") are
issued solely by the protocol lead; contributor documents use "proposal / transcript /
pending review" only.

## Context hygiene clause (binding for all executing agents)
Execution context may only contain: (1) the frozen registry and its referenced
originals; (2) the current task batch; (3) canonical documents (charter, audit
reports, work orders). Philosophy volumes, historical design drafts, unreviewed
hypotheses, and the pathogen archive (any material containing previously falsified
claims) must never enter execution context. Legacy code is read-only reference:
port with attribution (`# ported-from: <path>`), never bulk-copy.

## Engineering discipline
- No TODO / placeholder / mock-only fake completion.
- Where a mock is unavoidable, mark it: `# MOCK-SCOPE: <scope> | 转真条件: <condition>`.
- All randomness takes explicit seeds; experiment artifacts carry data fingerprints.
- Evidence sections quote real run output only; the phrase "expected output" is
  banned from any evidence field.
- Self-assigned perfect scores trigger mandatory independent review.
- Math-signed deliverables follow the R3 v3 process: a hand-worked self-check proves
  understanding; machine recomputation certifies the digits; both required to freeze.
