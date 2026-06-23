"""The One · complete form — one first-class engine that composes the de-risked native pieces
into a single perceive -> identify -> verify-do -> credential loop.

This is the architectural收口 of the B-line: instead of scattered probes, ONE object that runs
the whole native causal-cognition loop and returns ONE credential:

  1. PERCEIVE      — a latent confounder observed only as a noisy stream is integrated by the
                     SSM perception front-end into a recovered confounder (B3/perception).
  2. IDENTIFY      — among pre-treatment candidates, pick the confounders by association with
                     both treatment and outcome (the data-driven half of identification;
                     post-treatment colliders are excluded by the standard non-descendant
                     assumption — a declared boundary, since pure observation is Markov-limited).
  3. VERIFY-DO     — the native verifiable engine computes do() on the identified set with a
                     replay-checked derivation chain + three-zone honest status (B1/B4).
  4. CREDENTIAL    — one unified, recomputable credential (or an honest abstain).

Honest scope: discrete/continuous toy-to-mid SCMs, the regimes each piece was de-risked in.
Fluent language is NOT part of this loop (B2 open frontier). The value is the COMPOSITION:
every native conclusion carries perception provenance + identification rationale + a verified
do() credential, end to end.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from theone.native.engine import NativeVerifiableEngine
from theone.native.perception import SSMPerception


@dataclass
class CompleteFormResult:
    effect: Optional[float]
    zone: str
    trustworthy: bool
    confounders: list
    perception: Optional[str]
    credential: dict

    def __str__(self) -> str:
        e = "abstain" if self.effect is None else f"{self.effect:+.4f}"
        return f"<CompleteForm do={e} zone={self.zone} adjust_on={self.confounders}>"


class CompleteForm:
    """One native engine: perceive -> identify -> verifiable do -> credential."""

    def __init__(self, assoc_threshold: float = 0.1) -> None:
        self.engine = NativeVerifiableEngine()
        self.assoc_threshold = assoc_threshold

    # --- identification: data-driven confounder selection among pre-treatment candidates ---
    @staticmethod
    def _assoc(df: pd.DataFrame, a: str, b: str) -> float:
        if df[a].nunique() < 2 or df[b].nunique() < 2:
            return 0.0
        return abs(float(np.corrcoef(df[a], df[b])[0, 1]))

    def identify(self, df: pd.DataFrame, treatment: str, outcome: str,
                 pre_treatment: list[str]) -> list[str]:
        """A confounder is associated with BOTH treatment and outcome; irrelevants with neither.
        post-treatment colliders are excluded by passing only pre-treatment candidates."""
        return sorted(v for v in pre_treatment
                      if self._assoc(df, v, treatment) > self.assoc_threshold
                      and self._assoc(df, v, outcome) > self.assoc_threshold)

    # --- the full loop ---------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, treatment: str = "X", outcome: str = "Y",
                pre_treatment: Optional[list[str]] = None,
                streams: Optional[np.ndarray] = None, n_strata: int = 2) -> CompleteFormResult:
        """Run perceive(optional) -> identify -> verify-do -> credential.

        If `streams` is given, the confounder is PERCEIVED from the stream (added as column 'U')
        and used as the adjustment variable. Otherwise the confounder set is identified from the
        observed pre-treatment candidates in `df`."""
        df = df.copy()
        perception = None
        if streams is not None:
            u_hat = SSMPerception().to_confounder(streams, n_strata=n_strata)
            df["U"] = u_hat
            confounders = ["U"]
            perception = f"SSM-perceived confounder from {streams.shape[1]}-step streams"
        else:
            cands = pre_treatment if pre_treatment is not None else \
                [c for c in df.columns if c not in (treatment, outcome)]
            confounders = self.identify(df, treatment, outcome, cands)

        conf = confounders[0] if confounders else None
        r = self.engine.estimate(df, treatment=treatment, outcome=outcome, confounder=conf)
        cred = {
            "claim": r.credential.get("claim"),
            "zone": r.zone,
            "regime": "complete-form: perceive -> identify -> verify-do -> credential",
            "perception": perception,
            "identified_confounders": confounders,
            "e_value": r.e_value,
            "structural_stability": r.structural_stability,
            "replay_ok": r.replay_ok,
            "chain_hash": r.credential.get("chain_hash"),
            # cross-pillar fields: make the conclusion storable in sovereign causal memory
            # (indexed by its de-confounded signature, not surface text).
            "treatment": treatment, "target": outcome,
            "adjustment_set": confounders, "effect": r.effect,
        }
        return CompleteFormResult(
            effect=r.effect, zone=r.zone, trustworthy=r.is_trustworthy(),
            confounders=confounders, perception=perception, credential=cred)

    # --- act: turn a verified conclusion into a CREDENTIALED decision (or honest abstain) ---
    def recommend(self, result: "CompleteFormResult", min_effect: float = 0.05) -> dict:
        """The 'act' step of the cognitive loop. Only a TRUSTWORTHY (VERIFIABLE + replay-ok)
        conclusion with a materially non-zero effect yields an action; otherwise the system
        ABSTAINS with a reason — it never acts on an unverified or fragile belief."""
        cred = result.credential
        if not result.trustworthy:
            return {"action": "abstain", "reason": f"conclusion not trustworthy (zone={result.zone}, "
                    f"replay_ok={cred.get('replay_ok')}) — refusing to act on an unverified belief.",
                    "credential": cred}
        if result.effect is None or abs(result.effect) < min_effect:
            return {"action": "abstain", "reason": f"effect {result.effect} is below the action "
                    f"threshold {min_effect} — too small to act on.", "credential": cred}
        direction = "apply the treatment" if result.effect > 0 else "withhold the treatment"
        return {"action": "recommend", "decision": direction,
                "because": f"verified causal effect {result.effect:+.3f} (zone {result.zone}, "
                           f"E-value {cred.get('e_value')}), adjusting for {result.confounders}.",
                "credential": cred}

    # --- continuous-outcome path (real-world outcomes are rarely binary) --------------------
    def analyze_continuous(self, X: np.ndarray, t: np.ndarray, yf: np.ndarray,
                           covariate_sufficient: bool = True,
                           streams: Optional[np.ndarray] = None) -> CompleteFormResult:
        """Same complete-form loop for CONTINUOUS outcomes: optional perception -> learned
        adjustment (TARNet over the covariates) -> reproducible-replay + continuous three-zone
        -> one credential. Adjustment is over the whole covariate block (TARNet), so the
        'confounders' here are 'all covariates'."""
        X = np.asarray(X, np.float32)
        perception = None
        if streams is not None:
            feats = SSMPerception().perceive_features(streams, k=4)
            X = feats if X.size == 0 else np.concatenate([X, feats], axis=1)
            perception = f"SSM-perceived {feats.shape[1]} covariates from {streams.shape[1]}-step streams"
        r = self.engine.estimate_continuous(X, np.asarray(t, np.float32),
                                            np.asarray(yf, np.float32),
                                            covariate_sufficient=covariate_sufficient)
        cred = {
            "claim": r.credential.get("claim"),
            "zone": r.zone,
            "regime": "complete-form continuous: perceive -> learned adjustment -> verify-do",
            "perception": perception,
            "e_value": r.e_value,
            "reproducibility_stability": r.structural_stability,
            "replay_ok": r.replay_ok,
            "chain_hash": r.credential.get("chain_hash"),
            # cross-pillar fields so a continuous conclusion is storable in sovereign memory too
            "treatment": "T", "target": "Y_continuous",
            "adjustment_set": ["<all covariates>"], "effect": r.effect,
        }
        return CompleteFormResult(
            effect=r.effect, zone=r.zone, trustworthy=r.is_trustworthy(),
            confounders=["<all covariates>"], perception=perception, credential=cred)


__all__ = ["CompleteForm", "CompleteFormResult"]
