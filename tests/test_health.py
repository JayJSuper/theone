"""Tests for the med-interaction checker (offline; a stub LLM is injected)."""
from __future__ import annotations
from theone.app.health import HealthChecker
from theone.layer1_perception.llm_client import LLMReply
from theone.domains import drug_interactions as di


class _StubLLM:
    def __init__(self, reply):
        self._r = reply
    def available(self):
        return True
    def chat(self, prompt, system=None):
        return LLMReply(self._r, "stub", live=True)


def test_kb_known_interaction():
    v = di.check("华法林", "阿司匹林")
    assert v.status is di.Status.KNOWN_INTERACTION and v.severity == "严重"


def test_kb_unknown_pair_abstains():
    v = di.check("美托洛尔", "阿莫西林")
    assert v.status is di.Status.UNKNOWN


def test_alias_resolution_chinese_and_english():
    assert di.extract_drugs("我吃伟哥和硝酸甘油") == ["sildenafil", "nitroglycerin"]


def test_checker_flags_danger_with_basis():
    r = HealthChecker(llm=None).check("华法林和阿司匹林一起吃吗")
    assert r["badge"] == "danger" and "冲突" in r["headline"]
    assert "basis" in r and "action" in r and r["disclaimer"]


def test_checker_abstains_on_unknown():
    r = HealthChecker(llm=None).check("美托洛尔和阿莫西林")
    assert r["badge"] == "abstain" and "没有" in r["headline"]


def test_checker_needs_two_drugs():
    r = HealthChecker(llm=None).check("我在吃阿司匹林")
    assert r["badge"] == "ask"


def test_guard_catches_dangerous_false_reassurance():
    # stub LLM confidently says SAFE for a known-dangerous pair -> red catch
    app = HealthChecker(llm=_StubLLM("这两种药一起吃完全安全,没问题。"))
    r = app.check("华法林和阿司匹林能一起吃吗")
    assert r["badge"] == "danger"
    assert "🚩" in r.get("guard", "") and r["llm_stance"] == "safe"


def test_guard_catches_guess_on_unknown():
    # stub LLM gives a confident verdict on a pair with no verified data -> caught
    app = HealthChecker(llm=_StubLLM("一起吃是安全的,没有相互作用。"))
    r = app.check("美托洛尔和阿莫西林一起吃吗")
    assert r["badge"] == "abstain" and "🚩" in r.get("guard", "")


def test_classifier_not_fooled_by_negated_safe():
    # "不安全" must be read as risky, not safe (no false catch)
    app = HealthChecker(llm=_StubLLM("同时服用会增加出血风险,因此不安全。"))
    r = app.check("华法林和阿司匹林")
    assert r["llm_stance"] == "risky"
    assert "🚩" not in r.get("guard", "")   # AI was correct -> no false accusation
