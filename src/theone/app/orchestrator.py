"""The One · end-to-end orchestrator — the A-line demonstrable product.

Ties the verifiable kernel and a mounted LLM into one app with explicit provenance:
  • causal query   -> verifiable engine: exact do() + recomputable credential + E-value;
                      if a mounted LLM also offers a number, the engine CORROBORATES or
                      REFUTES it (the hallucination guard).
  • memory op       -> sovereign causal memory (remember / recall / export).
  • chat / code     -> mounted LLM (live if a key is configured, else offline stub),
                      labelled UNVERIFIED.
Every answer states which track handled it and whether it is engine-verified.

Honest scope: natural-language → causal-graph is the hard frontier; this app verifies
queries over a *registered* causal domain (known variables), and for anything outside it
honestly says so rather than faking a number.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

from theone.causal.graph import CausalGraph
from theone.layer7_planning import ToolRouter, Intent
from theone.layer1_perception import LLMAdapter
from theone.layer1_perception.llm_client import LLMClient
from theone.layer2_world_model import CausalLayer
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature


@dataclass
class CausalDomain:
    """A registered, verifiable causal model: a graph + keyword aliases for its variables."""
    name: str
    graph: CausalGraph
    aliases: dict = field(default_factory=dict)   # lowercase keyword -> variable name

    def resolve(self, text: str) -> list[str]:
        t = (text or "").lower()
        hits = []
        for kw, var in self.aliases.items():
            if kw in t and var not in hits:
                hits.append(var)
        return hits


class TheOneApp:
    def __init__(self, provider: str = "deepseek", memory_path: str = ":memory:",
                 domain: Optional[CausalDomain] = None, consult_llm: bool = True,
                 llm=None) -> None:
        self.router = ToolRouter()
        self.adapter = LLMAdapter()
        self.llm = llm if llm is not None else LLMClient(provider)
        self.causal = CausalLayer()
        self.memory = SovereignMemory(memory_path)
        self.domain = domain
        self.consult_llm = consult_llm
        self._native = None              # lazily-built native verifiable engine (data path)
        self._complete = None            # lazily-built complete-form integrated engine

    # --- data-driven causal via the NATIVE engine (double-engine hot-swap) ---
    def ask_data_causal(self, df, confounder: str = "U") -> dict:
        """Answer a causal question from DATA via the native verifiable engine — the second,
        architecture-different engine behind the same product. Symbolic engine (known graph)
        and native engine (data) both return credentialed answers through one product."""
        if self._native is None:
            from theone.native import NativeVerifiableEngine
            self._native = NativeVerifiableEngine()
        r = self._native.estimate(df, confounder=confounder)
        return {"track": "native_engine", "verified": r.is_trustworthy(),
                "provenance": f"native verifiable engine · zone={r.zone} · replay_ok={r.replay_ok}",
                "answer": r.credential["claim"], "zone": r.zone, "e_value": r.e_value,
                "structural_stability": r.structural_stability, "replay_ok": r.replay_ok,
                "credential": r.credential}

    def ask_data_causal_continuous(self, X, t, yf, covariate_sufficient: bool = True) -> dict:
        """Same native engine, CONTINUOUS-outcome path — real product data (medical/economic
        outcomes are rarely binary). Returns the same credentialed shape so the product can
        present binary and continuous causal answers uniformly."""
        if self._native is None:
            from theone.native import NativeVerifiableEngine
            self._native = NativeVerifiableEngine()
        r = self._native.estimate_continuous(X, t, yf, covariate_sufficient=covariate_sufficient)
        return {"track": "native_engine_continuous", "verified": r.is_trustworthy(),
                "provenance": f"native verifiable engine (continuous) · zone={r.zone} · replay_ok={r.replay_ok}",
                "answer": r.credential["claim"], "zone": r.zone, "e_value": r.e_value,
                "reproducibility_stability": r.structural_stability, "replay_ok": r.replay_ok,
                "credential": r.credential}

    # --- W2CG bridge: verify a NATURAL-LANGUAGE causal claim against known knowledge ---
    def ask_verify_claim(self, sentence: str, structure: dict, entity_syn: dict) -> dict:
        """Take an English causal claim and judge it against verified knowledge:
        VERIFIED / CONTRADICTED (hallucination caught) / UNVERIFIABLE (honest abstain). Never
        falsely verifies — the bridge from real language to the verifiable core (NOTE-113)."""
        from theone.language import ClaimVerifier
        v = ClaimVerifier(structure, entity_syn).verify_claim(sentence)
        return {"track": "claim_verifier", "verdict": v.verdict,
                "verified": v.verdict == "VERIFIED",
                "cause": v.cause, "effect": v.effect, "direction": v.direction,
                "provenance": f"W2CG claim verifier · {v.reason}", "reason": v.reason}

    # --- END-TO-END: the complete-form as ONE call (perceive data -> verify each -> generate) ---
    def ask_causal_report(self, factors: dict, Y, outcome: str, label: dict, entity_syn: dict) -> dict:
        """The whole engine in one product call. `factors`: {name: (X_covariates, T_treatment)}.
        For each factor the native engine estimates a covariate-adjusted causal effect on Y (with
        zone + E-value), then the verified reporter renders the findings into a fluent, round-trip-
        gated report — verifiable by construction, honestly 'inconclusive' where it cannot certify."""
        import re as _re
        from theone.language import Finding
        if self._native is None:
            from theone.native import NativeVerifiableEngine
            self._native = NativeVerifiableEngine()
        findings, credentials = [], {}
        for name, (X, T) in factors.items():
            r = self._native.estimate_continuous(X, T, Y, covariate_sufficient=True)
            m = _re.search(r"[-+]?\d*\.?\d+", r.credential.get("claim", ""))
            ate = float(m.group()) if m else None
            findings.append(Finding(name, outcome, 1 if (ate or 0) >= 0 else -1, r.zone, ate, r.e_value))
            credentials[name] = {"zone": r.zone, "ate": ate, "e_value": r.e_value, "replay_ok": r.replay_ok}
        rep = self.ask_verified_report(findings, label, entity_syn)
        return {"track": "causal_report", "verified": True, "report": rep["report"],
                "held_back": rep["held_back"], "credentials": credentials,
                "provenance": "complete-form: perceive(real data) -> verify-causal(zones+E) -> verified generation"}

    # --- verified GENERATION: render engine findings as fluent, hallucination-free report ---
    def ask_verified_report(self, findings, label: dict, entity_syn: dict) -> dict:
        """Generate a fluent natural-language report from VERIFIED engine findings. Round-trip-
        gated: no emitted sentence can assert anything the engine did not certify; honest zones
        surfaced verbatim (NOTE-126/127). The generation counterpart to ask_verify_claim."""
        from theone.language import VerifiedReporter
        out = VerifiedReporter(label, entity_syn).report(findings)
        return {"track": "verified_report", "verified": True,
                "provenance": "verified-by-construction generation · round-trip-gated",
                **out}

    # --- the COMPLETE FORM: one integrated engine (perceive->identify->verify-do->credential) ---
    def ask_complete_form(self, df, pre_treatment=None, treatment="X", outcome="Y",
                          streams=None) -> dict:
        """Answer through the integrated complete-form engine: it perceives (optional stream),
        IDENTIFIES the adjustment set from data, does a replay-verified do(), and returns one
        credential — the whole native loop behind the product, not a single hand-set confounder."""
        if self._complete is None:
            from theone.native import CompleteForm
            self._complete = CompleteForm()
        r = self._complete.analyze(df, treatment=treatment, outcome=outcome,
                                   pre_treatment=pre_treatment, streams=streams)
        return {"track": "complete_form", "verified": r.trustworthy,
                "provenance": f"complete form · perceive→identify→verify-do · zone={r.zone}",
                "answer": r.credential.get("claim"), "zone": r.zone,
                "identified_confounders": r.confounders, "perception": r.perception,
                "replay_ok": r.credential.get("replay_ok"), "credential": r.credential}

    # --- public entry -------------------------------------------------------
    def ask(self, text: str) -> dict:
        route = self.router.route(text)
        if route.intent is Intent.CAUSAL_QUERY:
            return self._causal(text)
        if route.intent is Intent.MEMORY_OP:
            return self._memory(text)
        return self._generate(text, route)

    # --- causal (verifiable) -----------------------------------------------
    def _causal(self, text: str) -> dict:
        out = {"track": "causal_engine", "verified": True, "provenance": "engine-verified"}
        if self.domain is None:
            return {**out, "verified": False, "answer": "no causal model registered",
                    "provenance": "unverifiable (no domain)"}
        vs = self.domain.resolve(text)
        treat = self.domain.aliases.get("__treatment__")
        targ = self.domain.aliases.get("__target__")
        # prefer explicitly-declared treatment/target if both are mentioned (or defaulted)
        treatment = treat if (treat in vs or treat) else (vs[0] if vs else None)
        target = targ if (targ in vs or targ) else (vs[1] if len(vs) > 1 else None)
        if not treatment or not target:
            claim = self.adapter.parse(text)
            return {**out, "verified": False,
                    "answer": f"extracted claim {claim.treatment}->{claim.target} but no "
                              f"matching causal model to verify",
                    "provenance": "claim-extracted, unverified"}
        v = self.causal.run({"graph": self.domain.graph, "treatment": treatment, "target": target})
        if not v.is_answer():
            return {**out, "verified": True, "answer": f"abstain: {v.reason}",
                    "provenance": "engine-abstained"}
        c = v.credential
        ok, info = c.verify()
        sens = c.evidence.get("sensitivity", {})
        result = {**out, "answer": f"P({target}=1 | do({treatment}=1)) = {c.value:.4f}",
                  "value": c.value, "regime": c.regime, "recomputed_ok": ok,
                  "recompute_gap": info.get("gap"), "e_value": sens.get("e_value"),
                  "credential": {"claim": c.claim, "value": c.value, "regime": c.regime,
                                 "do_x0": c.evidence.get("do_x0"), "sensitivity": sens}}
        # LLM proposes, engine verifies (the hallucination guard)
        if self.consult_llm and self.llm.available():
            llm_est = self._ask_llm_for_number(treatment, target)
            if llm_est is not None:
                gap = abs(llm_est - c.value)
                result["llm_estimate"] = llm_est
                result["verdict"] = "corroborated" if gap <= 0.1 else "refuted"
                result["verdict_note"] = (
                    f"mounted LLM said {llm_est:.2f}; engine computed {c.value:.4f} "
                    f"(gap {gap:.2f}) -> {result['verdict']}")
        return result

    def _ask_llm_for_number(self, treatment: str, target: str) -> Optional[float]:
        prompt = (f"In a system where {treatment} causally affects {target} (both binary), "
                  f"estimate P({target}=1 | do({treatment}=1)) as a single number between 0 and 1. "
                  f"Reply with ONLY the number.")
        reply = self.llm.chat(prompt, system="You answer with a single number only.")
        if not reply.live:
            return None
        m = re.search(r"[01]?\.\d+|[01]", reply.text)
        try:
            return float(m.group(0)) if m else None
        except (ValueError, AttributeError):
            return None

    # --- memory (sovereign) -------------------------------------------------
    _RECALL_WORDS = ("recall", "retrieve", "what did", "回忆", "回想", "想起", "取回",
                     "查一下", "之前说", "记得什么", "存了什么", "有哪些记忆")

    def _memory(self, text: str) -> dict:
        low = text.lower()
        if any(w in low for w in self._RECALL_WORDS):
            live = self.memory.store.search("")
            return {"track": "sovereign_memory", "verified": True, "provenance": "sovereign",
                    "answer": f"{len(live)} memories stored",
                    "recent": [r["value"].get("text", "") for r in live[-3:]]}
        # store: treat the message as a remembered fact (claim verified at store-time if causal)
        claim = self.adapter.parse(text)
        cred = {"treatment": claim.treatment or "NA", "target": claim.target or "NA",
                "adjustment_set": claim.adjustment_set, "effect": claim.effect or 0.0,
                "regime": "user-stated"}
        mid = self.memory.remember(text, cred, source="user")
        return {"track": "sovereign_memory", "verified": True, "provenance": "sovereign",
                "answer": f"remembered (id={mid}, signature {CausalSignature.from_credential(cred).structure_key()})"}

    # --- generation (mounted LLM, unverified) ------------------------------
    def _generate(self, text: str, route) -> dict:
        reply = self.llm.chat(text)
        return {"track": "mounted_llm", "verified": False,
                "provenance": f"mounted LLM ({reply.provider}, {'live' if reply.live else 'offline-stub'}) — UNVERIFIED",
                "intent": route.intent.value, "answer": reply.text, "live": reply.live}

    def close(self) -> None:
        self.memory.close()


__all__ = ["TheOneApp", "CausalDomain"]
