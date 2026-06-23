"""L2 StructureFitLayer — bridge a discovered skeleton + expert orientation into a
do()-computable CausalGraph, on the spine.

It consumes `discovered_edges` (from CausalDiscoveryLayer, data-supported skeleton) and
`oriented_edges` (expert / interventional orientation). It ANSWERs only if the expert
orientation's skeleton MATCHES the data-discovered skeleton (else the orientation
contradicts the data -> ABSTAIN), then fits Laplace-smoothed CPTs and emits the graph.
The credential's value is the graph content hash, recomputed by re-fitting (deterministic).
"""
from __future__ import annotations
from typing import Any
import pandas as pd

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer2_world_model.fit import fit_cpts
from theone.layer2_world_model.discovery import _skeleton


class StructureFitLayer(CredentialedLayer):
    name = "L2f_structure_fit"
    layer_index = 2

    def process(self, inputs: Any) -> LayerVerdict:
        data = inputs["data"]
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        oriented = [tuple(e) for e in inputs["oriented_edges"]]
        discovered = inputs.get("discovered_edges")

        if discovered is not None and _skeleton(oriented) != _skeleton(discovered):
            return LayerVerdict.abstain(
                self.name, "expert orientation skeleton contradicts the data-discovered "
                           "skeleton (orientation not supported by data)")
        try:
            g = fit_cpts(df, oriented)
        except Exception as e:
            return LayerVerdict.abstain(self.name, f"CPT fit failed: {e}")

        cred = Credential(
            self.name, claim="CPTs fit on the (data-consistent, expert-oriented) structure",
            value=g.content_hash(),
            regime="CPTs MLE-fit (Laplace-smoothed); orientation expert-sourced; structure-assumed",
            recompute=lambda: fit_cpts(df, oriented).content_hash(),
            tolerance=0.0,
            evidence={"oriented_edges": [list(e) for e in oriented], "n": int(len(df))})
        return LayerVerdict.answer(self.name, cred, value={**inputs, "graph": g})


__all__ = ["StructureFitLayer"]
