"""Minimal orchestrator v0. CC-11.
# MOCK-SCOPE: no LLM attached - rule-based routing v0 (regex on query syntax).
# 转真条件: Demo 3 PromptRouter attaches real open-weight base models.
Credentials are REAL computed artifacts (graph hash recomputable), not decoration."""
from __future__ import annotations
import re
import time
from dataclasses import dataclass, field
from ..types import TheOneConfig
from ..causal.engine import InterventionEngine
from ..causal.identify import find_adjustment_set
from ..memory.store import MemoryStore
from .. import __version__

_OBS = re.compile(r"P\(\s*(\w+)\s*=\s*(\d+)\s*\|\s*(\w+)\s*=\s*(\d+)\s*\)")
_DO = re.compile(r"P\(\s*(\w+)\s*=\s*(\d+)\s*\|\s*do\(\s*(\w+)\s*=\s*(\d+)\s*\)\s*\)")
_REMEMBER = re.compile(r"^remember\s+(\S+)\s*=\s*(.+)$", re.I)
_RECALL = re.compile(r"^recall\s+(\S+)$", re.I)


@dataclass
class AgentResponse:
    answer: str
    credential: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, engine: InterventionEngine, memory: MemoryStore,
                 config: TheOneConfig | None = None) -> None:
        self.engine, self.memory = engine, memory
        self.config = config or TheOneConfig()

    def _credential(self, query: str, method: str, extra: dict) -> dict:
        cred = {"query": query, "method": method,
                "graph_hash": self.engine.g.content_hash(),
                "timestamp": time.time(), "engine_version": __version__}
        cred.update(extra)
        return cred

    def handle(self, request: str) -> AgentResponse:
        m = _DO.search(request)
        if m:
            t, tv, dv, dval = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            r = self.engine.query_intervention(t, tv, {dv: dval})
            adj = find_adjustment_set(self.engine.g, dv, t)
            return AgentResponse(
                answer=f"P({t}={tv}|do({dv}={dval})) = {r.value:.6f}",
                credential=self._credential(request, r.method,
                                            {"adjustment_set": sorted(adj) if adj is not None else None}))
        m = _OBS.search(request)
        if m:
            t, tv, gv, gval = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            r = self.engine.query_observation(t, tv, {gv: gval})
            return AgentResponse(
                answer=f"P({t}={tv}|{gv}={gval}) = {r.value:.6f}",
                credential=self._credential(request, r.method, {"adjustment_set": None}))
        m = _REMEMBER.match(request.strip())
        if m:
            mid = self.memory.put(m.group(1), m.group(2).strip(), source="user")
            return AgentResponse(answer=f"stored #{mid}: {m.group(1)}",
                                 credential=self._credential(request, "memory_put",
                                                             {"memory_id": mid}))
        m = _RECALL.match(request.strip())
        if m:
            hits = self.memory.search(m.group(1))
            ans = "; ".join(
                f"#{h['id']} {h['key']}={h['value']}" for h in hits) or "no memory found"
            return AgentResponse(answer=ans,
                                 credential=self._credential(request, "memory_search",
                                                             {"hits": len(hits)}))
        return AgentResponse(
            answer="unrecognized request (v0 understands: P(Y=1|X=1), P(Y=1|do(X=1)), remember k=v, recall k)",
            credential=self._credential(request, "unrouted", {}))
