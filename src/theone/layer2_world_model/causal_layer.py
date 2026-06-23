"""L2 CausalLayer — the verified InterventionEngine wrapped as a CredentialedLayer.

Re-homing, not reimplementation: the do() value comes from the frozen engine; the
credential's recompute is the independent pgmpy IPRG. If the two disagree beyond
1e-6 the spine auto-downgrades the answer to ABSTAIN. The regime field carries the
honest, frozen limit (NOTE-004): the computation is exact, the *structure* is assumed.
"""
from __future__ import annotations
from typing import Any

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.causal.engine import InterventionEngine
from theone.layer2_world_model.iprg import pgmpy_do1
from theone.layer2_world_model.sensitivity import e_value_for_do


class CausalLayer(CredentialedLayer):
    name = "L2_causal"
    layer_index = 2

    def process(self, inputs: Any) -> LayerVerdict:
        """inputs: {graph: CausalGraph, treatment: str, target: str}.
        Emits P(target=1 | do(treatment=1)) with an IPRG-recomputable credential."""
        g = inputs["graph"]
        x, y = inputs["treatment"], inputs["target"]
        try:
            engine = InterventionEngine(g)
        except Exception as e:  # invalid graph / CPT → admissibility failure
            return LayerVerdict.abstain(self.name, f"graph invalid: {e}")
        try:
            do_value = round(float(engine.query_intervention(y, 1, {x: 1}).value), 12)
            do_value0 = round(float(engine.query_intervention(y, 1, {x: 0}).value), 12)
        except ZeroDivisionError as e:
            return LayerVerdict.abstain(self.name, f"degenerate query: {e}")

        # quantify the part the structure-assumed limit cannot certify: how strong an
        # unmeasured confounder would have to be to overturn this contrast (E-value).
        sens = e_value_for_do(do_value, do_value0)

        cred = Credential(
            self.name,
            claim=f"P({y}=1 | do({x}=1)) under the assumed structure",
            value=do_value,
            regime="computation-exact, structure-assumed (NOTE-004)",
            recompute=lambda: round(float(pgmpy_do1(g, x, y)), 12),
            tolerance=1e-6,
            evidence={"graph_hash": g.content_hash(), "method": "graph_surgery_do",
                      "adjustment": "prior-weighted confounders (graph surgery)",
                      "do_x0": do_value0, "sensitivity": sens},
        )
        return LayerVerdict.answer(self.name, cred, value={**inputs, "do_value": do_value})
