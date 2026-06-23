"""Finance calculation checker — exact computation that catches an LLM's arithmetic slips.

Computes the answer exactly (recomputable from the stated formula), then asks the mounted
LLM the same question for a single number and flags a meaningful discrepancy.

Not financial advice — it computes standard formulas and shows them so anyone can check.
"""
from __future__ import annotations
import re
from theone.domains import finance_calc as fc

DISCLAIMER = "本工具按标准公式精确计算并给出公式供你核对,不是投资/借贷建议。"

_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)")


def _extract_number(text: str):
    nums = [float(m.replace(",", "")) for m in _NUM.findall(text or "")]
    # the headline money figure is usually the largest plausible number
    nums = [n for n in nums if n >= 1]
    return max(nums) if nums else None


class FinanceChecker:
    def __init__(self, llm=None, consult_llm: bool = True) -> None:
        self.llm = llm
        self.consult_llm = consult_llm

    def check(self, text: str) -> dict:
        calc = fc.parse_and_compute(text)
        if calc is None:
            return {"badge": "ask", "headline": "请把数字说全",
                    "detail": "比如:'100万房贷,利率4.9%,30年,月供多少?' 或 '10万年化6%复利20年到期多少?'",
                    "disclaimer": DISCLAIMER}
        res = {"badge": "ok", "headline": f"✓ 精确计算:{calc.value:,.2f} {calc.unit}",
               "detail": calc.plain, "basis": f"公式(你可自行复算):{calc.basis}",
               "breakdown": calc.breakdown, "disclaimer": DISCLAIMER, "value": calc.value}

        if self.consult_llm and self.llm is not None and self.llm.available():
            ask = self._number_prompt(calc)
            reply = self.llm.chat(ask, system="只回答一个金额数字,不要解释、不要单位文字。")
            if reply.live:
                llm_num = _extract_number(reply.text)
                res["llm_said"] = reply.text.strip()[:80]
                if llm_num is not None:
                    rel = abs(llm_num - calc.value) / max(calc.value, 1e-9)
                    res["llm_value"] = llm_num
                    if rel > 0.01:   # >1% off = a real arithmetic error
                        res["guard"] = (f"🚩 注意:刚才那个普通 AI 算成了 ¥{llm_num:,.2f},"
                                        f"但精确值是 ¥{calc.value:,.2f},差了 ¥{abs(llm_num-calc.value):,.2f}。"
                                        f"财务上这种小数点的偏差会被放大——这就是为什么要用可复算的精确计算。")
                    else:
                        res["guard"] = (f"✓ 这次那个 AI 也算对了(¥{llm_num:,.2f})。"
                                        f"但我给的是带公式、可被任何人复算的精确值。")
        return res

    @staticmethod
    def _number_prompt(calc: fc.Calc) -> str:
        b = calc.breakdown
        if calc.kind == "mortgage":
            return (f"贷款本金 {b['principal']:.0f} 元,年利率 {b['annual_rate']*100:.2f}%,"
                    f"{b['years']:.0f} 年,等额本息。每月还款是多少元?只回答数字。")
        if calc.kind == "compound":
            return (f"本金 {b['principal']:.0f} 元,年化 {b['annual_rate']*100:.2f}% 复利,"
                    f"{b['years']:.0f} 年后到期总额是多少元?只回答数字。")
        return (f"本金 {b['principal']:.0f} 元,年利率 {b['annual_rate']*100:.2f}% 单利,"
                f"{b['years']:.0f} 年利息是多少元?只回答数字。")


__all__ = ["FinanceChecker", "DISCLAIMER"]
