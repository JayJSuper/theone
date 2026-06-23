"""Tests for the finance calculation verifier (offline; stub LLM injected)."""
from __future__ import annotations
from theone.domains import finance_calc as fc
from theone.app.finance import FinanceChecker
from theone.layer1_perception.llm_client import LLMReply


class _StubLLM:
    def __init__(self, reply):
        self._r = reply
    def available(self):
        return True
    def chat(self, prompt, system=None):
        return LLMReply(self._r, "stub", live=True)


def test_mortgage_exact():
    c = fc.parse_and_compute("100万房贷,利率4.9%,30年,月供多少?")
    assert c.kind == "mortgage"
    assert abs(c.value - 5307.27) < 0.5         # standard amortization
    assert c.breakdown["months"] == 360


def test_compound_exact():
    c = fc.parse_and_compute("10万年化6%复利20年到期多少?")
    assert c.kind == "compound"
    assert abs(c.value - 100000 * 1.06 ** 20) < 1.0


def test_simple_interest():
    c = fc.parse_and_compute("5万单利年利率3%存4年利息多少")
    assert c.kind == "simple" and abs(c.value - 50000 * 0.03 * 4) < 1e-6


def test_parse_needs_full_numbers():
    assert fc.parse_and_compute("房贷月供多少") is None


def test_checker_returns_exact_with_formula():
    r = FinanceChecker(llm=None).check("100万房贷,利率4.9%,30年,月供多少?")
    assert r["badge"] == "ok" and "精确计算" in r["headline"]
    assert "公式" in r["basis"] and r["value"] > 0


def test_guard_catches_llm_arithmetic_error():
    # stub LLM returns a wrong number -> red catch
    r = FinanceChecker(llm=_StubLLM("6000")).check("100万房贷,利率4.9%,30年,月供多少?")
    assert "🚩" in r.get("guard", "")
    assert r["llm_value"] == 6000.0


def test_guard_confirms_when_llm_correct():
    r = FinanceChecker(llm=_StubLLM("5307.27")).check("100万房贷,利率4.9%,30年,月供多少?")
    assert "🚩" not in r.get("guard", "") and "也算对了" in r["guard"]
