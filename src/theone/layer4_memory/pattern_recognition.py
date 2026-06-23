"""L4 · causal pattern recognition — find causal structures that RECUR across the
sovereign memory's live entries. Recurrence is computed on the de-confounded causal
signatures (not surface text), so a pattern is a genuinely repeated causal relationship,
not a textual coincidence.
"""
from __future__ import annotations
from collections import Counter

from theone.memory.signature import CausalSignature


class PatternRecognizer:
    def __init__(self, memory) -> None:
        self.mem = memory

    def _live_signatures(self) -> list[CausalSignature]:
        return [CausalSignature.from_dict(r["value"]["signature"]) for r in self.mem._all_live()]

    def frequent_edges(self, min_support: int = 2) -> list[dict]:
        """Recurring (treatment -> target) edges across memories."""
        sigs = self._live_signatures()
        total = max(len(sigs), 1)
        c = Counter((s.treatment, s.target) for s in sigs)
        return [{"edge": list(e), "count": n, "support": round(n / total, 3)}
                for e, n in c.most_common() if n >= min_support]

    def frequent_structures(self, min_support: int = 2) -> list[dict]:
        """Recurring full structures (treatment->target | adjustment | regime)."""
        sigs = self._live_signatures()
        total = max(len(sigs), 1)
        c = Counter(s.structure_key() for s in sigs)
        return [{"structure": k, "count": n, "support": round(n / total, 3)}
                for k, n in c.most_common() if n >= min_support]


__all__ = ["PatternRecognizer"]
