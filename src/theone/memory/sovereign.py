"""Sovereign causal memory — pillar 2, end to end.

Ties together the three things that make this *the One's* memory rather than a vector
store: (1) memories are indexed by **causal signature** (de-confounded fingerprint
inherited from a computation-pillar credential), so retrieval-for-a-decision is immune
to the surface confounding that fools embedding search; (2) memories are **versioned**
— a revision supersedes but never erases its predecessor, so the belief history is
auditable; (3) the store is **sovereign** — every memory has mandatory provenance,
delete means gone, and export is the take-your-data right (inherited from MemoryStore).

This module is deliberately small and dependency-light: real SQLite persistence, real
versioning, a real signature index. It is *not* a distributed vector DB — it is the
minimal honest realisation of the pillar's claim.
"""
from __future__ import annotations
import time
from dataclasses import dataclass

from .store import MemoryStore
from .signature import CausalSignature
from . import retrieval as R
import numpy as np


@dataclass
class Recall:
    mem_id: int
    text: str
    signature: CausalSignature
    version: int
    score: float          # lower = closer (signature distance) for credentialed recall


class SovereignMemory:
    """A causal memory the user owns. Backed by a persistent MemoryStore."""

    def __init__(self, path: str) -> None:
        self.store = MemoryStore(path)

    # --- write --------------------------------------------------------------
    def remember(self, text: str, credential: dict, source: str,
                 embedding: list | None = None) -> int:
        """Store a memory indexed by the causal signature *derived from its
        credential*. The de-confounding is inherited, not recomputed."""
        sig = CausalSignature.from_credential(credential)
        value = {"text": text, "signature": sig.to_dict(), "version": 1,
                 "supersedes": None, "embedding": embedding}
        return self.store.put(sig.structure_key(), value, source=source)

    def revise(self, mem_id: int, text: str, credential: dict, source: str,
               embedding: list | None = None) -> int:
        """Versioned update: the new memory supersedes the old; the old is retained
        (auditable belief history), never silently overwritten."""
        prev = self.store.get(mem_id)
        if prev is None:
            raise KeyError(f"no memory {mem_id} to revise")
        sig = CausalSignature.from_credential(credential)
        value = {"text": text, "signature": sig.to_dict(),
                 "version": int(prev["value"].get("version", 1)) + 1,
                 "supersedes": mem_id, "embedding": embedding}
        return self.store.put(sig.structure_key(), value, source=source)

    # --- read (the whole point) ---------------------------------------------
    def _all_live(self) -> list:
        """Live memories = those not superseded by a later version."""
        rows = self.store.search("")
        superseded = {r["value"].get("supersedes") for r in rows
                      if r["value"].get("supersedes") is not None}
        return [r for r in rows if r["id"] not in superseded]

    def recall_for_decision(self, query: CausalSignature, k: int = 1,
                            effect_weight: float = 1.0) -> list[Recall]:
        """CREDENTIALED retrieval: rank by *causal-signature* distance, not surface
        text. Under confounding this returns the memory whose de-confounded effect
        actually matches the decision — where embedding search returns the text
        look-alike with the wrong effect."""
        live = self._all_live()
        scored = []
        for r in live:
            sig = CausalSignature.from_dict(r["value"]["signature"])
            d = query.distance(sig, effect_weight=effect_weight)
            scored.append(Recall(r["id"], r["value"]["text"], sig,
                                 r["value"].get("version", 1), d))
        scored.sort(key=lambda x: x.score)
        return scored[:k]

    def recall_by_surface(self, query_embedding: np.ndarray, k: int = 1,
                          kernel: str = "cosine") -> list[Recall]:
        """BASELINE: flat embedding retrieval (what a vector store does). Ranks by
        surface similarity — and so transfers the wrong effect under confounding.
        Provided to make the pillar's advantage measurable, not as the product."""
        live = [r for r in self._all_live() if r["value"].get("embedding") is not None]
        if not live:
            return []
        keys = np.array([r["value"]["embedding"] for r in live], dtype=float)
        scores = R.retrieve(np.asarray(query_embedding, dtype=float), keys, kernel=kernel)
        order = list(np.argsort(-scores))[:k]
        out = []
        for i in order:
            r = live[i]
            sig = CausalSignature.from_dict(r["value"]["signature"])
            out.append(Recall(r["id"], r["value"]["text"], sig,
                              r["value"].get("version", 1), float(scores[i])))
        return out

    # --- sovereignty --------------------------------------------------------
    def history(self, mem_id: int) -> list:
        """The version chain ending at mem_id (auditable belief history)."""
        chain, cur = [], self.store.get(mem_id)
        while cur is not None:
            chain.append(cur)
            sup = cur["value"].get("supersedes")
            cur = self.store.get(sup) if sup is not None else None
        return list(reversed(chain))

    def export(self) -> str:
        return self.store.export()

    def forget(self, mem_id: int) -> bool:
        """Sovereign delete: gone means gone."""
        return self.store.delete(mem_id)

    def close(self) -> None:
        self.store.close()
