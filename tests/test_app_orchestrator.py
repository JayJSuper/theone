"""Tests for the A-line product orchestrator (offline — a stub LLM is injected)."""
from __future__ import annotations
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.app import TheOneApp, CausalDomain
from theone.layer1_perception.llm_client import LLMReply


class _StubLLM:
    """Deterministic offline LLM: no network; returns a fixed number / echo."""
    def __init__(self, number="0.50"):
        self._n = number
    def available(self):
        return True
    def chat(self, prompt, system=None):
        if "single number" in (system or "") or "ONLY the number" in prompt:
            return LLMReply(self._n, "stub", live=True)
        return LLMReply(f"stub-reply: {prompt[:40]}", "stub", live=True)


def _domain():
    g = CausalGraph()
    for n in ("S", "T", "R"):
        g.add_variable(Variable(n))
    g.add_edge("S", "T"); g.add_edge("S", "R"); g.add_edge("T", "R")
    g.set_cpt("S", {(): {0: 0.6, 1: 0.4}})
    g.set_cpt("T", {(0,): {0: 0.7, 1: 0.3}, (1,): {0: 0.3, 1: 0.7}})
    oR = list(g.parent_order("R"))
    vals = {(0, 0): .80, (0, 1): .90, (1, 0): .30, (1, 1): .55}
    g.set_cpt("R", {tuple(s if p == "S" else t for p in oR): {1: v, 0: round(1 - v, 2)}
                    for (s, t), v in vals.items()})
    return CausalDomain("recovery", g,
                        {"treatment": "T", "recovery": "R", "__treatment__": "T", "__target__": "R"})


def _app(number="0.50"):
    return TheOneApp(domain=_domain(), llm=_StubLLM(number), memory_path=":memory:")


def test_causal_query_is_engine_verified_with_credential():
    app = _app()
    r = app.ask("what is the effect of the treatment on recovery?")
    assert r["track"] == "causal_engine" and r["verified"] is True
    assert r["recomputed_ok"] is True and r["recompute_gap"] == 0.0
    assert r["e_value"] is not None and r["value"] > 0
    app.close()


def test_hallucination_guard_refutes_wrong_llm_number():
    app = _app(number="0.50")          # engine computes 0.76 -> refuted
    r = app.ask("effect of treatment on recovery?")
    assert r["verdict"] == "refuted"
    app2 = _app(number="0.76")          # matches engine -> corroborated
    r2 = app2.ask("effect of treatment on recovery?")
    assert r2["verdict"] == "corroborated"
    app.close(); app2.close()


def test_memory_store_and_recall_sovereign():
    app = _app()
    s = app.ask("remember that the launch is delayed to november")
    assert s["track"] == "sovereign_memory" and "remembered" in s["answer"]
    rec = app.ask("recall what we stored")
    assert rec["track"] == "sovereign_memory" and "1 memories" in rec["answer"]
    app.close()


def test_generation_is_labelled_unverified():
    app = _app()
    r = app.ask("write a python function to add two numbers")
    assert r["track"] == "mounted_llm" and r["verified"] is False
    assert "UNVERIFIED" in r["provenance"]
    app.close()


def test_no_domain_is_honestly_unverifiable():
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    r = app.ask("what is the causal effect of X on Y?")
    assert r["verified"] is False and "no causal model" in r["answer"]
    app.close()


def test_native_engine_path_double_engine():
    """The product can answer a causal question via the NATIVE engine (data path)."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0); n = 6000
    u = (rng.random(n) < 0.45).astype(int)
    x = (rng.random(n) < np.where(u == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .25, (0, 1): .55, (1, 0): .65, (1, 1): .95}
    y = np.array([1 if rng.random() < py[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    df = pd.DataFrame({"U": u, "X": x, "Y": y})
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    r = app.ask_data_causal(df, confounder="U")
    assert r["track"] == "native_engine" and r["replay_ok"] is True
    assert r["zone"] in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED", "REJECT")
    app.close()


def test_w2cg_claim_verification_product_path():
    """The product can verify a natural-language causal claim: verify / contradict / abstain,
    never falsely verifying."""
    struct = {("smoking", "cancer"): {"direction": 1, "magnitude": 2},
              ("drug", "recovery"): {"direction": 0, "magnitude": None}}
    syn = {"smoking": ["smoking", "cigarette", "lighting up", "puffing"],
           "cancer": ["cancer", "tumor"], "drug": ["drug", "pill"],
           "recovery": ["recovery", "getting better"]}
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    assert app.ask_verify_claim("Lighting up raises tumor risk.", struct, syn)["verdict"] == "VERIFIED"
    assert app.ask_verify_claim("Smoking decreases cancer.", struct, syn)["verdict"] == "CONTRADICTED"
    assert app.ask_verify_claim("Coffee boosts mood.", struct, syn)["verdict"] == "UNVERIFIABLE"
    app.close()


def test_causal_report_end_to_end_product_path():
    """One product call runs the engine per factor and emits a verified, round-trip-gated report —
    the complete-form end to end (perceive data -> verify-causal -> verified generation)."""
    import numpy as np
    rng = np.random.default_rng(0); n = 1200
    X = rng.normal(size=(n, 5)).astype(np.float32)
    T = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * X[:, 0])))).astype(np.float32)
    Y = (1.5 + X[:, 0] + 2.5 * T + rng.normal(scale=0.5, size=n)).astype(np.float32)
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    out = app.ask_causal_report({"delay": (X, T)}, Y, "default",
                                {"delay": "a recent missed payment", "default": "default"},
                                {"delay": ["missed payment", "a recent missed payment"], "default": ["default"]})
    assert out["track"] == "causal_report"
    assert "delay" in out["credentials"] and out["credentials"]["delay"]["replay_ok"]
    assert isinstance(out["report"], list)           # a (possibly empty) set of verifiable sentences
    app.close()


def test_verified_report_generation_product_path():
    """The product can GENERATE a fluent report from engine findings, round-trip-gated so no
    sentence asserts more than the engine certified, and honest 'inconclusive' for REJECT."""
    from theone.language import Finding
    label = {"payment_delay": "a recent missed payment", "default": "default"}
    syn = {"payment_delay": ["missed payment", "a recent missed payment", "delinquency"],
           "default": ["default", "defaulting"]}
    findings = [
        Finding("payment_delay", "default", +1, "VERIFIABLE", ate=0.34, e_value=3.7),
        Finding("payment_delay", "default", +1, "REJECT"),
    ]
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    r = app.ask_verified_report(findings, label, syn)
    assert r["track"] == "verified_report" and r["verifiable_by_construction"]
    assert r["n_emitted"] == 2                      # the verified one + the honest inconclusive one
    assert any("verified" in s for s in r["report"]) and any("inconclusive" in s for s in r["report"])
    app.close()


def test_complete_form_product_path():
    """The product can answer via the integrated complete-form engine (identify + verify-do)."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(0); n = 6000
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .40, (0, 1): .62, (1, 0): .70, (1, 1): .92}
    Y = np.array([1 if rng.random() < py[(x, u)] else 0 for x, u in zip(X, U)])
    Z = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    out = app.ask_complete_form(df, pre_treatment=["U", "Z"])
    assert out["track"] == "complete_form" and out["replay_ok"] is True
    assert out["identified_confounders"] == ["U"]       # identifies confounder, drops irrelevant Z
    app.close()


def test_native_engine_continuous_product_path():
    """The product can answer a CONTINUOUS-outcome causal question via the native engine."""
    import numpy as np
    rng = np.random.default_rng(0); n = 1500
    X = rng.normal(size=(n, 6)).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * X[:, 0] - 0.5 * X[:, 1])))).astype(np.float32)
    y = (2.0 + X[:, 0] + 0.5 * X[:, 2] + 3.0 * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    app = TheOneApp(domain=None, llm=_StubLLM(), memory_path=":memory:")
    r = app.ask_data_causal_continuous(X, t, y, covariate_sufficient=True)
    assert r["track"] == "native_engine_continuous" and r["replay_ok"] is True
    assert r["zone"] in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED", "REJECT")
    assert "reproducibility_stability" in r
    app.close()
