"""A-line product acceptance — exercise every scenario end-to-end (offline) and print a
PASS/FAIL dashboard. No API key required: the mounted-LLM consult degrades to a stub, but
the verifiable scenarios (health KB, finance exact compute, sovereign memory) run fully.

Usage:  .venv/bin/python scripts/verify_product.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from theone.app.product import TheOneProduct                     # noqa: E402
from theone.layer1_perception.llm_client import LLMReply         # noqa: E402


class _Stub:
    def available(self):
        return False
    def chat(self, prompt, system=None):
        return LLMReply("stub", "stub", live=False)


CASES = [
    # (question, expected scenario, expected badge, must-be-verified)
    ("华法林和阿司匹林能一起吃吗?", "health", "danger", True),
    ("阿莫西林和美托洛尔一起吃可以吗?", "health", "abstain", True),
    ("二甲双胍和赖诺普利一起吃呢?", "health", "ok", True),
    ("100万房贷利率4.9%30年月供多少?", "finance", "ok", True),
    ("10万年化6%复利20年到期多少?", "finance", "ok", True),
    ("记住下周三复诊", "memory", "ok", True),
    ("回忆一下", "memory", "ok", True),
    ("今天适合穿什么?", "chat", "chat", False),
]


def main():
    print("=" * 60)
    print("The One · A-line product acceptance")
    print("=" * 60)
    p = TheOneProduct(memory_path=":memory:", llm=_Stub())
    ok_all = True
    for q, exp_sc, exp_badge, exp_ver in CASES:
        r = p.ask(q)
        ok = (r.get("scenario") == exp_sc and r.get("badge") == exp_badge
              and bool(r.get("verified")) == exp_ver)
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {exp_sc:8} {q[:24]:26} -> "
              f"{r.get('scenario')}/{r.get('badge')} verified={r.get('verified')}")

    # sovereign history
    hist_ok = len(p.export_history()) == len(CASES)
    cleared = p.clear_history()
    hist_ok &= (cleared == len(CASES) and p.export_history() == [])
    ok_all &= hist_ok
    print(f"  [{'PASS' if hist_ok else 'FAIL'}] sovereign history: logged {len(CASES)}, "
          f"exported, cleared {cleared}")
    p.close()
    print("-" * 60)
    print(f"OVERALL: {'ALL GREEN' if ok_all else 'FAILURES PRESENT'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
