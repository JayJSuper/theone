"""Health med-interaction checker — the felt-value demo, in plain language.

Flow for "我同时吃 A 和 B,有问题吗?":
  1. extract the two drugs;
  2. check the VERIFIED knowledge base (flag known interaction / low-concern / abstain);
  3. ask the mounted LLM the same question and classify its stance;
  4. if the LLM's confident answer contradicts the verified basis (e.g. says "safe" for a
     known-dangerous pair, or asserts a verdict on an unknown pair), CATCH it.

Everything is plain-language and defers to a doctor/pharmacist. The system never tells
anyone to take, stop, or change a medication.
"""
from __future__ import annotations
import re
from theone.domains import drug_interactions as di

DISCLAIMER = "本工具只核验公开的用药冲突知识,不是医疗建议;任何用药决定请咨询医生或药师。"


def _classify_llm(text: str) -> str:
    """Classify the mounted LLM's stance: safe / risky / unsure.
    Order matters: an explicit risk / negated-safe ('不安全') must win over a bare '安全'
    substring, so we never mis-read a correct warning as reassurance."""
    t = (text or "")
    # 1) reassurance about NO risk / no interaction -> safe (checked first to beat '风险')
    if re.search(r"没有(明显|已知|重大|特殊)?(相互作用|风险)|无(明显)?风险|"
                 r"不会有(明显)?风险|没有.{0,3}冲突", t):
        return "safe"
    # 2) explicit risk or negated safety -> risky
    if re.search(r"不安全|不可以|不能(一起|同时)|不建议|避免|禁忌|危险|出血|风险|严重|"
                 r"横纹肌|肌肉损伤|肝损伤|危及|需.{0,3}监测|慎用", t):
        return "risky"
    # 3) explicit reassurance, not negated
    if re.search(r"(通常|一般)?(是)?安全|没问题|可以(一起|同时|放心)|相对安全", t):
        return "safe"
    if re.search(r"不确定|无法确定|不清楚|咨询|请教|不知道|视情况|因人而异|需更多信息", t):
        return "unsure"
    return "unsure"


class HealthChecker:
    def __init__(self, llm=None, consult_llm: bool = True) -> None:
        self.llm = llm
        self.consult_llm = consult_llm

    def check(self, text: str) -> dict:
        drugs = di.extract_drugs(text)
        if len(drugs) < 2:
            return {"badge": "ask", "headline": "请告诉我具体的两种药名",
                    "detail": "比如:'我同时吃华法林和阿司匹林,会有问题吗?'",
                    "disclaimer": DISCLAIMER, "drugs": drugs}
        a, b = drugs[0], drugs[1]
        v = di.check(a, b)

        if v.status is di.Status.KNOWN_INTERACTION:
            res = {"badge": "danger", "headline": f"⚠️ 已核实:{a} + {b} 有{v.severity}冲突",
                   "detail": v.plain, "basis": f"依据:{v.basis}",
                   "action": "请尽快咨询医生或药师,不要自行调整用药。"}
        elif v.status is di.Status.KNOWN_LOW_CONCERN:
            res = {"badge": "ok", "headline": f"✅ 已核实:{a} + {b} 无重大已知冲突",
                   "detail": v.plain, "basis": f"依据:{v.basis}",
                   "action": "仍建议向药师确认你的具体情况。"}
        else:
            res = {"badge": "abstain", "headline": f"🤷 我没有 {a} + {b} 的可靠数据",
                   "detail": v.plain, "basis": "诚实告诉你:这超出我已核实的范围。",
                   "action": "别只信我、也别只信普通 AI——这种事请直接问药师。"}
        res.update({"disclaimer": DISCLAIMER, "drugs": [a, b], "status": v.status.value})

        # ask the mounted LLM and catch dangerous disagreement
        if self.consult_llm and self.llm is not None and self.llm.available():
            reply = self.llm.chat(
                f"病人同时服用{a}和{b},把它们一起吃安全吗?用一句话回答,说明安全还是有风险。",
                system="你是谨慎的用药信息助手,一句话作答。")
            if reply.live:
                stance = _classify_llm(reply.text)
                res["llm_said"] = reply.text.strip()[:140]
                res["llm_stance"] = stance
                if v.status is di.Status.KNOWN_INTERACTION and stance == "safe":
                    res["guard"] = ("🚩 注意:刚才那个普通 AI 说'安全'——但这其实是有据可查的"
                                    f"{v.severity}冲突。这就是为什么不能只信 AI 的一句话。")
                elif v.status is di.Status.UNKNOWN and stance in ("safe", "risky"):
                    res["guard"] = ("🚩 注意:那个普通 AI 给了一个肯定的结论,但这个组合其实"
                                    "没有可靠依据可查——它在拿你的健康猜。我选择不猜。")
                elif v.status is di.Status.KNOWN_INTERACTION and stance == "risky":
                    res["guard"] = ("✓ 这次那个 AI 也判断有风险,和核实一致——但它给不出可核实的"
                                    "依据,而我能。这正是区别。")
                elif stance == "unsure":
                    res["guard"] = "(这次那个 AI 比较谨慎、没乱下结论。)"
        return res


__all__ = ["HealthChecker", "DISCLAIMER"]
