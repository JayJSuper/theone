"""TheOneProduct — the unified A-line product.

One entry point that auto-routes a question to the right verifiable scenario, keeps a
sovereign query history the user owns (export / clear), and labels every answer's
provenance:
  • drug-interaction question  -> verified KB (flag / abstain / catch LLM)
  • financial calculation       -> exact computation (recomputable, catch LLM's slips)
  • memory op                   -> sovereign memory
  • anything else               -> mounted LLM, labelled UNVERIFIED

Scenarios share one mounted LLM and one sovereign memory. Adding a scenario = adding a
detector + a checker; the kernel discipline (verify-or-abstain + provenance) is shared.
"""
from __future__ import annotations
import re

from theone.layer1_perception.llm_client import LLMClient
from theone.app.health import HealthChecker
from theone.app.finance import FinanceChecker
from theone.domains import drug_interactions as di
from theone.domains import finance_calc as fc
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature

_MEM_STORE = re.compile(r"记住|记下|记一下|存一下|帮我记")
_MEM_RECALL = re.compile(r"回忆|回想|想起|查一下记|我存了|有哪些记|历史")
_HEALTH_HINT = re.compile(r"一起吃|同时吃|同时服|相互作用|配伍|混着|一起服|能不能一起|一起用")


class TheOneProduct:
    SCENARIOS = ("health", "finance", "memory", "chat")

    def __init__(self, provider: str = "deepseek", memory_path: str = ":memory:",
                 llm=None) -> None:
        self.llm = llm if llm is not None else LLMClient(provider)
        self.health = HealthChecker(llm=self.llm)
        self.finance = FinanceChecker(llm=self.llm)
        self.memory = SovereignMemory(memory_path)
        self.history: list[dict] = []

    # --- routing ------------------------------------------------------------
    def detect(self, text: str) -> str:
        drugs = di.extract_drugs(text)
        if len(drugs) >= 2 or (drugs and _HEALTH_HINT.search(text or "")):
            return "health"
        if fc.parse_and_compute(text) is not None:
            return "finance"
        if _MEM_STORE.search(text or "") or _MEM_RECALL.search(text or ""):
            return "memory"
        return "chat"

    def ask(self, text: str) -> dict:
        scenario = self.detect(text)
        if scenario == "health":
            r = self.health.check(text); r["verified"] = r.get("badge") in ("danger", "ok", "abstain")
        elif scenario == "finance":
            r = self.finance.check(text); r["verified"] = True
        elif scenario == "memory":
            r = self._memory(text); r["verified"] = True
        else:
            r = self._chat(text); r["verified"] = False
        r["scenario"] = scenario
        self._log(text, r)
        return r

    # --- memory scenario ----------------------------------------------------
    def _memory(self, text: str) -> dict:
        if _MEM_RECALL.search(text or ""):
            rows = self.memory.store.search("")
            items = [rr["value"].get("text", "") for rr in rows]
            return {"badge": "ok", "headline": f"📒 你存了 {len(items)} 条记忆(完全属于你,可导出/删除)",
                    "detail": "\n".join(f"· {x}" for x in items[-6:]) or "(还没有记忆)",
                    "recent": items[-6:]}
        cred = {"treatment": "NA", "target": "NA", "adjustment_set": [],
                "effect": 0.0, "regime": "user-stated"}
        mid = self.memory.remember(text, cred, source="user")
        return {"badge": "ok", "headline": "📒 已记住(这条记忆属于你)",
                "detail": f"已存为第 {mid} 条,你随时可以导出或彻底删除。"}

    # --- chat (mounted LLM, unverified) ------------------------------------
    def _chat(self, text: str) -> dict:
        reply = self.llm.chat(text)
        return {"badge": "chat",
                "headline": "💬 这是普通 AI 的回答(我没有核验,仅供参考)",
                "detail": reply.text,
                "provenance": f"挂载 {reply.provider}({'在线' if reply.live else '离线'})· 未经核验"}

    # --- sovereign history --------------------------------------------------
    def _log(self, text: str, r: dict) -> None:
        self.history.append({"q": text, "scenario": r.get("scenario"),
                             "badge": r.get("badge"), "headline": r.get("headline"),
                             "verified": r.get("verified")})

    def export_history(self) -> list[dict]:
        return list(self.history)

    def clear_history(self) -> int:
        n = len(self.history)
        self.history.clear()
        return n

    def close(self) -> None:
        self.memory.close()


__all__ = ["TheOneProduct"]
