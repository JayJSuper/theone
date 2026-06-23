"""Language layer — the bridge between natural language and the verifiable causal core.

The One does not generate fluent open-domain text; it VERIFIES causal claims expressed in
natural language against what it can recompute. The ClaimVerifier parses an English causal
sentence into a structured claim (cause, effect, direction, magnitude) and judges it against
the engine's known causal structure: VERIFIED / CONTRADICTED / UNVERIFIABLE (honest abstain).
It NEVER falsely verifies — when unsure it abstains. This is the World-to-Causal-Graph bridge.
"""
from theone.language.claim_verifier import ClaimVerifier, Verdict
from theone.language.reporter import VerifiedReporter, Finding

__all__ = ["ClaimVerifier", "Verdict", "VerifiedReporter", "Finding"]
