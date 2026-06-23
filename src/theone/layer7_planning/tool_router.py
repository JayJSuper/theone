"""L7 · tool router — classify a user message and route it to the right handler.

The router decides WHICH track handles a request: a causal/interventional question goes
to the verifiable engine (L2, credentialed); a code/chat request goes to a mounted LLM
(A-line perception organ, unverified-but-labelled); a memory operation goes to sovereign
memory (L4). This is the dispatch that lets one system be a verifiable kernel AND mount
external LLMs — the routing is explicit, so every answer's provenance is known.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import re


class Intent(str, Enum):
    CAUSAL_QUERY = "causal_query"   # -> verifiable engine, credentialed
    MEMORY_OP = "memory_op"         # -> sovereign memory
    CODE = "code"                   # -> mounted LLM (labelled unverified)
    CHAT = "chat"                   # -> mounted LLM (labelled unverified)


@dataclass
class Route:
    intent: Intent
    handler: str          # logical handler name
    verifiable: bool      # whether the handler produces a recomputable credential


_CAUSAL = re.compile(
    r"\b(do\(|intervene|interven|causal|cause[ds]?|effect of|"
    r"counterfactual|if we set|adjust(ing)? for)\b"
    r"|因果|效应|效果|干预|反事实|导致|致使|影响|如果.{0,4}会",
    re.IGNORECASE)
_MEMORY = re.compile(
    r"\b(remember|recall|forget|memorize|memory|store this|what did .* say|retrieve)\b"
    r"|记住|记下|记一下|存一下|记忆|回忆|想起|回想|记得",
    re.IGNORECASE)
_CODE = re.compile(
    r"\b(write|implement|refactor|debug|fix)\b.*\b(code|function|class|script|program|bug)\b"
    r"|```|\bpython\b|\bjavascript\b"
    r"|(写|实现|编写|生成|改|调试).{0,6}(代码|函数|程序|脚本|方法|类)",
    re.IGNORECASE)


class ToolRouter:
    def route(self, text: str) -> Route:
        t = text or ""
        # explicit memory imperatives win even if the payload contains a causal claim
        # ("remember that X causes Y" is a store op; the claim is verified at store time)
        if _MEMORY.search(t):
            return Route(Intent.MEMORY_OP, "sovereign_memory", verifiable=True)
        if _CAUSAL.search(t):
            return Route(Intent.CAUSAL_QUERY, "causal_engine", verifiable=True)
        if _CODE.search(t):
            return Route(Intent.CODE, "mounted_llm", verifiable=False)
        return Route(Intent.CHAT, "mounted_llm", verifiable=False)


__all__ = ["Intent", "Route", "ToolRouter"]
