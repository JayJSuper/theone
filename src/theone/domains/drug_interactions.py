"""A small, VERIFIED drug-interaction knowledge base — the felt-value health demo.

The One's value here is NOT to give medical advice. It is to (1) flag a KNOWN, well-
documented interaction with a citable basis, or (2) honestly ABSTAIN when it has no
verified data — never guessing about your health — and (3) catch a mounted LLM that
confidently gives dangerous false reassurance or invents an interaction.

SAFETY: this is a tiny demo KB of textbook-level interactions, NOT comprehensive medical
software. Every answer defers to a doctor/pharmacist. The system never tells anyone to
take, stop, or change a medication.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Status(str, Enum):
    KNOWN_INTERACTION = "known_interaction"   # documented, flag it
    KNOWN_LOW_CONCERN = "known_low_concern"   # commonly co-prescribed, still defer
    UNKNOWN = "unknown"                        # not in verified KB -> ABSTAIN, do not guess


@dataclass
class Verdict:
    status: Status
    drug_a: str
    drug_b: str
    severity: str            # "严重" | "中等" | "低" | "未知"
    plain: str               # plain-language explanation (no jargon)
    basis: str               # the verifiable basis / source label


# canonical name -> aliases (Chinese + English, lowercased on lookup)
_ALIASES = {
    "warfarin": ["华法林", "warfarin", "可迈丁"],
    "aspirin": ["阿司匹林", "aspirin", "拜阿司匹林"],
    "ibuprofen": ["布洛芬", "ibuprofen", "芬必得"],
    "clarithromycin": ["克拉霉素", "clarithromycin"],
    "simvastatin": ["辛伐他汀", "simvastatin", "舒降之"],
    "atorvastatin": ["阿托伐他汀", "atorvastatin", "立普妥"],
    "sildenafil": ["西地那非", "sildenafil", "伟哥", "万艾可"],
    "nitroglycerin": ["硝酸甘油", "nitroglycerin", "硝酸酯", "消心痛", "isosorbide"],
    "metformin": ["二甲双胍", "metformin"],
    "lisinopril": ["赖诺普利", "lisinopril", "普利类", "ace抑制剂", "依那普利"],
    "potassium": ["钾", "补钾", "氯化钾", "potassium"],
    "ssri": ["ssri", "舍曲林", "氟西汀", "帕罗西汀", "百忧解"],
    "maoi": ["maoi", "单胺氧化酶抑制剂", "吗氯贝胺"],
    "metoprolol": ["美托洛尔", "metoprolol", "倍他乐克"],
    "amoxicillin": ["阿莫西林", "amoxicillin"],
}

# verified interactions: frozenset(pair) -> (severity, plain, basis)
_INTERACTIONS = {
    frozenset(("warfarin", "aspirin")): (
        "严重", "两种药都让血更难凝,一起用会明显增加出血风险(胃肠道、脑出血)。",
        "临床共识 / 药品说明书级别的已知相互作用"),
    frozenset(("warfarin", "ibuprofen")): (
        "严重", "布洛芬这类止痛药会加重华法林的出血风险,并可能伤胃。",
        "临床共识 / 说明书级别"),
    frozenset(("simvastatin", "clarithromycin")): (
        "严重", "克拉霉素会让辛伐他汀在体内堆积,可能引起严重肌肉损伤(横纹肌溶解)。",
        "CYP3A4 代谢抑制,说明书明确警示"),
    frozenset(("sildenafil", "nitroglycerin")): (
        "严重", "西地那非(伟哥)和硝酸甘油一起用会让血压骤降,可能危及生命。",
        "说明书绝对禁忌,广为人知"),
    frozenset(("lisinopril", "potassium")): (
        "中等", "普利类降压药本身会升钾,再额外补钾可能导致血钾过高、影响心脏。",
        "临床共识"),
    frozenset(("ssri", "maoi")): (
        "严重", "这两类抗抑郁药一起用可能引发'5-羟色胺综合征',很危险。",
        "精神科用药基本禁忌"),
}

# a few commonly co-prescribed pairs with no major known interaction (still defer)
_LOW_CONCERN = {
    frozenset(("metformin", "lisinopril")),
    frozenset(("amoxicillin", "metformin")),
    frozenset(("metoprolol", "aspirin")),
}


def _canon(text: str) -> list[str]:
    t = (text or "").lower()
    found = []
    for canon, aliases in _ALIASES.items():
        if any(a.lower() in t for a in aliases) and canon not in found:
            found.append(canon)
    return found


def extract_drugs(text: str) -> list[str]:
    """Return up to the canonical drug names mentioned in the text."""
    return _canon(text)


def check(drug_a: str, drug_b: str) -> Verdict:
    a, b = drug_a.lower(), drug_b.lower()
    # resolve to canonical if an alias was passed
    ca = _canon(a) or [a]
    cb = _canon(b) or [b]
    a, b = ca[0], cb[0]
    pair = frozenset((a, b))
    if pair in _INTERACTIONS:
        sev, plain, basis = _INTERACTIONS[pair]
        return Verdict(Status.KNOWN_INTERACTION, a, b, sev, plain, basis)
    if pair in _LOW_CONCERN:
        return Verdict(Status.KNOWN_LOW_CONCERN, a, b, "低",
                       "在已核实范围内,这两种药没有需要特别警示的相互作用——但这不代表绝对安全。",
                       "已核实无重大相互作用记录")
    return Verdict(Status.UNKNOWN, a, b, "未知",
                   "我没有这两种药相互作用的已核实数据。我不会拿你的健康去猜。",
                   "不在已核实知识库内")


def known_pairs() -> list:
    return [tuple(p) for p in _INTERACTIONS] + [tuple(p) for p in _LOW_CONCERN]


__all__ = ["Status", "Verdict", "check", "extract_drugs", "known_pairs"]
