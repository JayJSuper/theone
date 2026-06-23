"""Interactive REPL for the A-line product — talk to The One.

Routes each message: causal queries to the verifiable engine (with a recomputable
credential + hallucination guard against the mounted LLM), memory ops to sovereign
memory, chat/code to the mounted LLM (labelled UNVERIFIED).

Usage:  source ~/.theone_keys.env && .venv/bin/python scripts/theone_chat.py
Commands:  /help  /quit
Try:  "what is the effect of the treatment on recovery?"
      "remember the q3 launch slips to november"
      "recall"
      "write a python function to reverse a string"
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from theone.types import Variable                                    # noqa: E402
from theone.causal.graph import CausalGraph                          # noqa: E402
from theone.app import TheOneApp, CausalDomain                       # noqa: E402


def recovery_domain():
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
                        {"treatment": "T", "treat": "T", "recovery": "R", "recover": "R",
                         "severity": "S", "__treatment__": "T", "__target__": "R"})


def render(res: dict) -> None:
    mark = "✓ 已验证" if res["verified"] else "○ 未验证"
    print(f"\n  [{mark}] 来源: {res['provenance']}")
    print(f"  {res['answer']}")
    if res.get("e_value") is not None:
        print(f"  凭证: {res['regime']} · 复算 {'通过' if res['recomputed_ok'] else '失败'}"
              f"(gap {res['recompute_gap']:.1e}) · E-value {res['e_value']}")
    if res.get("verdict"):
        print(f"  幻觉护栏: {res['verdict_note']}")
    if res.get("recent"):
        for m in res["recent"]:
            print(f"    · {m}")
    print()


def main():
    app = TheOneApp(provider="deepseek", domain=recovery_domain())
    live = app.llm.available()
    print("=" * 64)
    print("  The One · 可验证认知内核(A 线演示)")
    print(f"  挂载 LLM: deepseek · {'在线' if live else '离线桩'} | 已注册因果域: recovery(T→R, 受 S 混杂)")
    print("  /help 查看示例 · /quit 退出")
    print("=" * 64)
    try:
        while True:
            try:
                text = input("\n你 › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in ("/quit", "/exit", "/q"):
                break
            if text == "/help":
                print(__doc__)
                continue
            render(app.ask(text))
    finally:
        app.close()
        print("\n已退出。记忆为会话内存(:memory:),退出即清。")


if __name__ == "__main__":
    main()
