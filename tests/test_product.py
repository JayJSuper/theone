"""Tests for the unified A-line product (offline; stub LLM injected)."""
from __future__ import annotations
import pytest
from theone.app.product import TheOneProduct
from theone.layer1_perception.llm_client import LLMReply


class _StubLLM:
    def available(self):
        return False           # offline -> no live consult; scenarios still work
    def chat(self, prompt, system=None):
        return LLMReply("stub", "stub", live=False)


def _prod():
    return TheOneProduct(memory_path=":memory:", llm=_StubLLM())


@pytest.mark.parametrize("text,scenario", [
    ("华法林和阿司匹林能一起吃吗?", "health"),
    ("布洛芬和华法林一起吃", "health"),
    ("100万房贷利率4.9%30年月供多少?", "finance"),
    ("10万年化6%复利20年到期多少?", "finance"),
    ("记住下周三复诊", "memory"),
    ("回忆一下", "memory"),
    ("今天适合穿什么?", "chat"),
])
def test_routing(text, scenario):
    assert _prod().detect(text) == scenario


def test_health_scenario_verified_flag():
    r = _prod().ask("华法林和阿司匹林一起吃吗")
    assert r["scenario"] == "health" and r["badge"] == "danger" and r["verified"] is True


def test_finance_scenario_exact():
    r = _prod().ask("100万房贷利率4.9%30年月供多少?")
    assert r["scenario"] == "finance" and r["verified"] is True and r["value"] > 0


def test_chat_scenario_unverified():
    r = _prod().ask("讲个笑话")
    assert r["scenario"] == "chat" and r["verified"] is False


def test_sovereign_history_accumulates_and_clears():
    p = _prod()
    p.ask("华法林和阿司匹林一起吃吗")
    p.ask("100万房贷利率4.9%30年月供多少?")
    p.ask("记住明天开会")
    assert len(p.export_history()) == 3
    assert p.clear_history() == 3 and p.export_history() == []
    p.close()
