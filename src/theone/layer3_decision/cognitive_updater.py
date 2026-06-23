"""L3 · cognitive updater — propose a causal-structure update only when new data
JUSTIFIES it by a recomputable BIC improvement.

When the world shifts, the held causal model may be wrong. This module re-discovers a
candidate structure on the new data and scores both the OLD and CANDIDATE structures
(over the same node set) with BIC. It proposes the update only if BIC improves beyond a
margin AND the skeleton actually changed — otherwise it abstains (keep the current
model; do not churn structure on noise). The credential carries the recomputable BIC
delta; the regime declares it remains subject to the L2 discovery limits (orientation /
latent confounding uncertified).
"""
from __future__ import annotations
import warnings
from typing import Any
import pandas as pd
# pgmpy 1.1.x deprecation FutureWarnings (caller-attributed); filter the specific
# messages before importing pgmpy.estimators (which emits StructureScore's at import).
for _msg in (r".*HillClimbSearch is deprecated.*", r".*StructureScore.*deprecated.*"):
    warnings.filterwarnings("ignore", message=_msg, category=FutureWarning)
from pgmpy.base import DAG
from pgmpy.estimators import BIC

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer2_world_model.discovery import discover, _skeleton


def _dag(nodes, edges) -> DAG:
    d = DAG()
    d.add_nodes_from(list(nodes))
    d.add_edges_from([tuple(e) for e in edges])
    return d


def bic_delta(df: pd.DataFrame, old_edges, new_edges) -> float:
    """BIC(new) - BIC(old) over the full node set on `df` (higher = new fits better)."""
    b = BIC(df.astype("category"))
    nodes = list(df.columns)
    return float(b.score(_dag(nodes, new_edges)) - b.score(_dag(nodes, old_edges)))


class CognitiveUpdater(CredentialedLayer):
    name = "L3u_cognitive_updater"
    layer_index = 3
    MARGIN = 10.0          # BIC difference > 10 = very strong evidence

    def process(self, inputs: Any) -> LayerVerdict:
        data = inputs["data"]
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        old_edges = [tuple(e) for e in inputs["old_edges"]]
        margin = float(inputs.get("margin", self.MARGIN))

        candidate = discover(df)
        changed = _skeleton(candidate) != _skeleton(old_edges)
        delta = bic_delta(df, old_edges, candidate)

        if not changed:
            return LayerVerdict.abstain(
                self.name, "re-discovered skeleton == current model; no structural change to propose")
        if delta <= margin:
            return LayerVerdict.abstain(
                self.name, f"BIC improvement {delta:.1f} <= margin {margin}; "
                           f"change not justified (do not churn structure on noise)")

        cred = Credential(
            self.name, claim="new data justifies a causal-structure update",
            value=round(delta, 6),
            regime=("structure-update proposed by BIC; remains structure-assumed and subject to "
                    "L2 discovery limits (orientation / latent confounding uncertified)"),
            recompute=lambda: round(bic_delta(df, old_edges, discover(df)), 6),
            tolerance=1e-6,
            evidence={"old_edges": [list(e) for e in old_edges],
                      "proposed_edges": [list(e) for e in candidate],
                      "bic_delta": round(delta, 3), "margin": margin})
        return LayerVerdict.answer(self.name, cred,
                                   value={**inputs, "proposed_edges": candidate, "bic_delta": delta})


__all__ = ["CognitiveUpdater", "bic_delta"]
