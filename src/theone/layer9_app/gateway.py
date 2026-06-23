"""L9 · application gateway — the outward interface that turns the verifiable kernel
into a service (the 'value-chain segment' deployment form).

The handlers are pure-python and JSON-serializable, so they are testable WITHOUT a web
framework. A thin optional FastAPI binding (`create_app`) is provided behind an import
guard — if fastapi is not installed, the handlers still work and tests still pass.

Every response carries provenance: which track handled it and whether the answer is
engine-verified (a recomputable credential) or came unverified from a mounted LLM.
"""
from __future__ import annotations
from typing import Any, Optional

from theone.layer7_planning import ToolRouter, Intent
from theone.layer2_world_model import CausalLayer
from theone.causal.graph import CausalGraph


class TheOneGateway:
    """Dispatch layer over the verifiable kernel. Stateless per request."""

    def __init__(self) -> None:
        self.router = ToolRouter()
        self.causal = CausalLayer()

    # --- routing / dialogue ------------------------------------------------
    def handle_message(self, text: str) -> dict:
        """Classify a message and report where it would be handled (provenance)."""
        r = self.router.route(text)
        return {"intent": r.intent.value, "handler": r.handler,
                "verifiable": r.verifiable, "text": text}

    # --- verifiable causal query ------------------------------------------
    def handle_causal(self, graph: CausalGraph, treatment: str, target: str) -> dict:
        """Run an interventional query through the credentialed L2 layer."""
        v = self.causal.run({"graph": graph, "treatment": treatment, "target": target})
        if not v.is_answer():
            return {"decision": "abstain", "reason": v.reason, "verifiable": True}
        c = v.credential
        ok, info = c.verify()
        return {"decision": "answer", "claim": c.claim, "value": c.value,
                "regime": c.regime, "recomputed_ok": ok, "recompute_info": info,
                "evidence": c.evidence, "verifiable": True}

    # --- credential verification (third-party endpoint) -------------------
    @staticmethod
    def verify_credential(value: Any, recomputed: Any, tolerance: float = 1e-6) -> dict:
        """A third party submits a claimed value + independent recompute; we check."""
        try:
            gap = abs(float(value) - float(recomputed))
            ok = gap <= tolerance
            return {"verified": ok, "gap": gap, "tolerance": tolerance}
        except (TypeError, ValueError):
            ok = value == recomputed
            return {"verified": ok, "gap": None, "tolerance": tolerance}

    def health(self) -> dict:
        return {"status": "ok", "kernel": "verifiable-causal", "tracks": ["mount-llm", "native-seed"]}


def create_app(gateway: Optional[TheOneGateway] = None):
    """Optional FastAPI binding. Returns a FastAPI app, or raises a clear error if
    fastapi is not installed (the pure-python handlers above do not need it)."""
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as e:  # pragma: no cover - exercised only when fastapi present
        raise RuntimeError(
            "FastAPI/pydantic not installed; the gateway handlers work without them — "
            "`pip install fastapi uvicorn pydantic` to expose HTTP endpoints.") from e

    gw = gateway or TheOneGateway()
    app = FastAPI(title="The One · verifiable causal kernel")

    class Msg(BaseModel):
        text: str

    @app.get("/health")
    def health():
        return gw.health()

    @app.post("/message")
    def message(m: Msg):
        return gw.handle_message(m.text)

    return app


__all__ = ["TheOneGateway", "create_app"]
