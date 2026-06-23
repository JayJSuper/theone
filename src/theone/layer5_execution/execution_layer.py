"""L5 ExecutionLayer — SafeExecutor wrapped as a CredentialedLayer.

Maps the SafeExecutor's EXECUTE/BLOCK/ABSTAIN decision onto the spine: only an
EXECUTE decision ANSWERs, carrying a credential whose recompute re-runs the
deterministic decision rule on the recorded checks (so the verdict is reproducible,
not asserted). Dangerous commands, sandbox escapes, and inadmissible causal drivers
all ABSTAIN — nothing touches the world without a recomputable green light.
"""
from __future__ import annotations
from typing import Any

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.execution.safe_executor import SafeExecutor


class ExecutionLayer(CredentialedLayer):
    name = "L5_execution"
    layer_index = 5

    def __init__(self, sandbox_root: str | None = None) -> None:
        self.ex = SafeExecutor(sandbox_root)

    def process(self, inputs: Any) -> LayerVerdict:
        kind = inputs.get("action_kind", "command")
        cc = inputs.get("causal_credential")
        if kind == "write":
            ec = self.ex.propose_write(inputs["target"], inputs.get("content", ""),
                                       causal_credential=cc,
                                       provenance=inputs.get("provenance", ""))
        else:
            ec = self.ex.propose_command(inputs["command"], causal_credential=cc,
                                         provenance=inputs.get("provenance", ""))

        if ec.decision != "EXECUTE":
            return LayerVerdict.abstain(self.name, f"{ec.decision}: {ec.reason}")

        # capture the recorded checks so recompute is deterministic and independent
        pc, cmdc, gate = ec.path_check, ec.command_check, ec.causal_gate
        cred = Credential(
            self.name, claim="action admissible (sandbox + causal gate)",
            value=ec.decision, regime="sandboxed dry-run",
            recompute=lambda: SafeExecutor._decide(pc, cmdc, gate)[0],
            tolerance=0.0, evidence=ec.as_record())
        return LayerVerdict.answer(self.name, cred, value={**inputs, "exec_credential": ec.as_record()})
