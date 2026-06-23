"""L4 MemoryLayer — SovereignMemory wrapped as a CredentialedLayer.

Operations: "remember" (write a memory indexed by its causal signature) and "recall"
(signature-distance retrieval for a decision). A recall ANSWERs only if a match
exists; the credential's recompute re-reads the persisted memory and re-derives its
structure key, catching any drift between what was returned and what is stored.
"""
from __future__ import annotations
from typing import Any

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.memory.sovereign import SovereignMemory
from theone.memory.signature import CausalSignature


class MemoryLayer(CredentialedLayer):
    name = "L4_memory"
    layer_index = 4

    def __init__(self, path: str = ":memory:") -> None:
        self.mem = SovereignMemory(path)

    def process(self, inputs: Any) -> LayerVerdict:
        op = inputs.get("op", "recall")
        if op == "remember":
            mem_id = self.mem.remember(inputs["text"], inputs["credential"],
                                       source=inputs.get("source", "fusion"),
                                       embedding=inputs.get("embedding"))
            sig = CausalSignature.from_credential(inputs["credential"])
            cred = Credential(
                self.name, claim="memory stored under its causal signature",
                value=sig.structure_key(), regime="signature-indexed write",
                recompute=lambda: CausalSignature.from_dict(
                    self.mem.store.get(mem_id)["value"]["signature"]).structure_key(),
                tolerance=0.0, evidence={"mem_id": mem_id})
            return LayerVerdict.answer(self.name, cred, value={**inputs, "mem_id": mem_id})

        # recall
        q = inputs["query_signature"]
        hits = self.mem.recall_for_decision(q, k=inputs.get("k", 1),
                                            effect_weight=inputs.get("effect_weight", 1.0))
        if not hits:
            return LayerVerdict.abstain(self.name, "no memory matches the causal signature")
        top = hits[0]
        threshold = inputs.get("match_threshold", 0.5)
        if top.score > threshold:   # admissibility gate: refuse a far causal-space match
            return LayerVerdict.abstain(
                self.name, f"nearest memory too far in causal space "
                           f"(score {top.score:.3f} > {threshold}); a confounded look-alike, not a match")
        cred = Credential(
            self.name, claim="memory retrieved by exact causal signature",
            value=top.signature.structure_key(), regime="signature-indexed retrieval",
            recompute=lambda: CausalSignature.from_dict(
                self.mem.store.get(top.mem_id)["value"]["signature"]).structure_key(),
            tolerance=0.0,
            evidence={"mem_id": top.mem_id, "score": top.score, "version": top.version})
        return LayerVerdict.answer(self.name, cred, value={**inputs, "recall": top})

    def close(self) -> None:
        self.mem.close()
