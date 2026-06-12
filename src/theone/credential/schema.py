"""Cognitive Credential schema v1 (proposal, T2-08).

A credential is the machine-checkable receipt attached to every important output
(charter ch.10: causal path / adjustment set / memory refs / seven-axis / hash).
This module is the FORMAL SPEC (a draft-07-style JSON Schema document) used for
publication and by the independent verifier. The generator (agent.orchestrator)
does NOT import this module: generation and verification stay decoupled so a
generator bug cannot mask an invalid credential (charter: separated deployment).

Status: proposal / pending review (contributor doc; not a frozen asset).
"""
from __future__ import annotations
import json

SCHEMA_VERSION = "credential/v1"

# methods the v0 orchestrator can emit (extend via amendment as routing grows)
ALLOWED_METHODS = (
    "graph_surgery_do",          # P(Y|do(X))
    "exact_joint_conditioning",  # P(Y|evidence)
    "memory_put",
    "memory_search",
    "unrouted",
)

# base fields every credential must carry
BASE_REQUIRED = ("query", "method", "graph_hash", "timestamp", "engine_version")

# extra fields each method must additionally carry
METHOD_REQUIRED = {
    "graph_surgery_do": ("adjustment_set",),
    "exact_joint_conditioning": ("adjustment_set",),
    "memory_put": ("memory_id",),
    "memory_search": ("hits",),
    "unrouted": (),
}

# reserved optional fields (charter ch.10 future surface), validated if present
OPTIONAL_FIELDS = ("value", "causal_path", "memory_refs", "confidence_seven_axis")

CREDENTIAL_SCHEMA_V1 = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://theone.dev/schema/credential/v1.json",
    "title": "The One Cognitive Credential v1",
    "type": "object",
    "required": list(BASE_REQUIRED),
    "additionalProperties": True,  # forward-compatible; verifier still range-checks knowns
    "properties": {
        "query": {"type": "string", "minLength": 1},
        "method": {"type": "string", "enum": list(ALLOWED_METHODS)},
        "graph_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "timestamp": {"type": "number", "exclusiveMinimum": 0},
        "engine_version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
        "adjustment_set": {
            "oneOf": [{"type": "null"},
                      {"type": "array", "items": {"type": "string"}}]},
        "memory_id": {"type": "integer", "minimum": 0},
        "hits": {"type": "integer", "minimum": 0},
        # reserved optional surface
        "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "causal_path": {"type": "array", "items": {"type": "string"}},
        "memory_refs": {"type": "array", "items": {"type": "integer"}},
        "confidence_seven_axis": {
            "type": "array", "minItems": 7, "maxItems": 7,
            "items": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    },
}


def schema_json(indent: int = 2) -> str:
    return json.dumps(CREDENTIAL_SCHEMA_V1, indent=indent)
