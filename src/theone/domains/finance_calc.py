"""Exact financial calculation — the second felt-value scenario.

LLMs frequently make arithmetic slips on multi-step financial math (mortgage payments,
compound growth). The One computes the answer EXACTLY (deterministic, recomputable by
anyone with the formula) and, when a mounted LLM offers a number, catches its error.

Supported, well-defined calculations (parsed from plain Chinese):
  • 等额本息房贷月供 (equal-installment mortgage monthly payment) + 总利息
  • 复利终值 (compound-interest future value)
  • 单利利息 (simple interest)

Honest scope: this computes the standard textbook formulas exactly; it is not financial
advice. Real loans have fees/compounding conventions that may differ — the result states
its formula so anyone can check the assumptions.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Calc:
    kind: str                 # "mortgage" | "compound" | "simple"
    value: float              # the headline number
    unit: str                 # "元/月" | "元" ...
    plain: str                # plain-language answer
    basis: str                # the exact formula (the recomputable credential)
    breakdown: dict           # inputs + key intermediate values


def _amount(text: str) -> Optional[float]:
    # 100万 / 50万元 / 20000元 / 2万 / 1.5万
    m = re.search(r"(\d+(?:\.\d+)?)\s*万", text)
    if m:
        return float(m.group(1)) * 10000
    m = re.search(r"(\d+(?:\.\d+)?)\s*元", text)
    if m:
        return float(m.group(1))
    return None


def _rate(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"利率\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _years(text: str) -> Optional[float]:
    m = re.search(r"(\d+(?:\.\d+)?)\s*年", text)
    if m:
        return float(m.group(1))
    return None


def parse_and_compute(text: str) -> Optional[Calc]:
    t = text or ""
    P, r, y = _amount(t), _rate(t), _years(t)

    # mortgage (equal installment)
    if re.search(r"月供|房贷|按揭|等额本息|每月还", t) and P and r is not None and y:
        c = r / 12.0
        N = int(round(y * 12))
        if c == 0:
            M = P / N
        else:
            M = P * c * (1 + c) ** N / ((1 + c) ** N - 1)
        total = M * N
        interest = total - P
        return Calc("mortgage", round(M, 2), "元/月",
                    f"贷款 {P/10000:.0f} 万、年利率 {r*100:.2f}%、{y:.0f} 年(等额本息):"
                    f"每月还 ¥{M:,.2f},总共还 ¥{total:,.2f},其中利息 ¥{interest:,.2f}。",
                    "等额本息公式 M = P·c·(1+c)^N / ((1+c)^N − 1),c=年利率/12,N=年数×12",
                    {"principal": P, "annual_rate": r, "years": y, "months": N,
                     "monthly_payment": round(M, 2), "total_paid": round(total, 2),
                     "total_interest": round(interest, 2)})

    # compound future value
    if re.search(r"复利|利滚利|年化|滚存|定投", t) and P and r is not None and y:
        FV = P * (1 + r) ** y
        return Calc("compound", round(FV, 2), "元",
                    f"本金 {P/10000:.2f} 万、年化 {r*100:.2f}%、{y:.0f} 年(复利):"
                    f"到期约 ¥{FV:,.2f},其中收益 ¥{FV-P:,.2f}。",
                    "复利终值 FV = P·(1+r)^年数",
                    {"principal": P, "annual_rate": r, "years": y,
                     "future_value": round(FV, 2), "gain": round(FV - P, 2)})

    # simple interest
    if re.search(r"单利|普通利息", t) and P and r is not None and y:
        interest = P * r * y
        return Calc("simple", round(interest, 2), "元",
                    f"本金 {P/10000:.2f} 万、年利率 {r*100:.2f}%、{y:.0f} 年(单利):"
                    f"利息 ¥{interest:,.2f},本息合计 ¥{P+interest:,.2f}。",
                    "单利 利息 = P·r·年数",
                    {"principal": P, "annual_rate": r, "years": y, "interest": round(interest, 2)})

    return None


__all__ = ["Calc", "parse_and_compute"]
