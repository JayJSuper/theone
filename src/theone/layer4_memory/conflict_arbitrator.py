"""L4 · conflict arbitrator — detect memories that make CONTRADICTORY causal claims
and propose a resolution. Two live memories conflict when they answer the SAME causal
question (same treatment -> target | adjustment | regime) with effects that differ
beyond a tolerance. Detection is on the de-confounded signature, so a real contradiction
is caught even when the texts differ — and a textual look-alike with a different
question is NOT flagged.

Resolution policy (proposals, never silent edits — sovereignty): prefer the newest
version; if versions tie, flag for human review. The arbitrator proposes; it does not
delete (the user owns the memory).
"""
from __future__ import annotations

from theone.memory.signature import CausalSignature


class ConflictArbitrator:
    def __init__(self, memory) -> None:
        self.mem = memory

    def find_conflicts(self, effect_tol: float = 0.1) -> list[dict]:
        rows = self.mem._all_live()
        groups: dict[str, list] = {}
        for r in rows:
            s = CausalSignature.from_dict(r["value"]["signature"])
            groups.setdefault(s.structure_key(), []).append(
                {"mem_id": r["id"], "effect": s.effect,
                 "version": int(r["value"].get("version", 1)), "text": r["value"].get("text", "")})

        conflicts = []
        for key, members in groups.items():
            effects = [m["effect"] for m in members]
            spread = max(effects) - min(effects)
            if len(members) > 1 and spread > effect_tol:
                newest = max(members, key=lambda m: (m["version"], m["mem_id"]))
                tie = sum(1 for m in members if m["version"] == newest["version"]) > 1
                resolution = ("flag for human review (version tie)" if tie
                              else f"keep newest (mem {newest['mem_id']}, v{newest['version']})")
                conflicts.append({"question": key, "effect_spread": round(spread, 4),
                                  "members": members, "resolution": resolution})
        return conflicts


__all__ = ["ConflictArbitrator"]
