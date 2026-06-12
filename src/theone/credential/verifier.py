"""Independent credential verifier (T2-08, proposal).

CHARTER INVARIANT: generation and verification are separately deployed with ZERO
shared logic. This module therefore:
  - imports NOTHING from agent.orchestrator / causal.* (the generators);
  - re-implements every check from first principles against the credential bytes;
  - never trusts a self-asserted field it can structurally check instead.

A self-contained hand-rolled validator (no jsonschema dependency). Returns a
structured verdict so a caller (or a separate audit service) can act on it.

Status: proposal / pending review (not a frozen asset).
"""
from __future__ import annotations
import re
import json
from .schema import (SCHEMA_VERSION, ALLOWED_METHODS, BASE_REQUIRED,
                     METHOD_REQUIRED, OPTIONAL_FIELDS)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _is_int(v) -> bool:                       # bool is a subclass of int - exclude it
    return isinstance(v, int) and not isinstance(v, bool)


def verify_credential(cred, expected_graph_hash: str | None = None) -> dict:
    """Verify one credential. Returns {valid, errors, checks}.

    expected_graph_hash (optional): a known-good hash the caller obtained
    independently; compared by STRING EQUALITY only (no hash logic shared with
    the generator). Mismatch => invalid (possible tamper / wrong graph)."""
    errors: list[str] = []

    if not isinstance(cred, dict):
        return {"valid": False, "errors": ["credential is not an object"], "checks": {}}

    # 1. base required fields present
    for f in BASE_REQUIRED:
        if f not in cred:
            errors.append(f"missing required field: {f}")

    # 2. base field types / formats
    if "query" in cred and not (isinstance(cred["query"], str) and cred["query"]):
        errors.append("query must be a non-empty string")
    method = cred.get("method")
    if method not in ALLOWED_METHODS:
        errors.append(f"method not in allowed set: {method!r}")
    gh = cred.get("graph_hash")
    if not (isinstance(gh, str) and _HEX64.match(gh)):
        errors.append("graph_hash must be 64 lowercase hex chars")
    ts = cred.get("timestamp")
    if not ((isinstance(ts, (int, float)) and not isinstance(ts, bool)) and ts > 0):
        errors.append("timestamp must be a positive number")
    ev = cred.get("engine_version")
    if not (isinstance(ev, str) and _SEMVER.match(ev)):
        errors.append("engine_version must be semver X.Y.Z")

    # 3. method-specific required fields + type rules
    for f in METHOD_REQUIRED.get(method, ()):  # empty tuple if method unknown
        if f not in cred:
            errors.append(f"method {method} requires field: {f}")
    if "adjustment_set" in cred:
        a = cred["adjustment_set"]
        if not (a is None or (isinstance(a, list) and all(isinstance(s, str) for s in a))):
            errors.append("adjustment_set must be null or a list of strings")
    if "memory_id" in cred and not _is_int(cred["memory_id"]) or \
            ("memory_id" in cred and cred["memory_id"] < 0):
        errors.append("memory_id must be a non-negative integer")
    if "hits" in cred and (not _is_int(cred["hits"]) or cred["hits"] < 0):
        errors.append("hits must be a non-negative integer")

    # 4. reserved optional fields (validated only if present)
    if "value" in cred:
        v = cred["value"]
        if not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0):
            errors.append("value must be a number in [0,1]")
    if "causal_path" in cred and not (
            isinstance(cred["causal_path"], list)
            and all(isinstance(s, str) for s in cred["causal_path"])):
        errors.append("causal_path must be a list of strings")
    if "memory_refs" in cred and not (
            isinstance(cred["memory_refs"], list)
            and all(_is_int(s) for s in cred["memory_refs"])):
        errors.append("memory_refs must be a list of integers")
    if "confidence_seven_axis" in cred:
        s = cred["confidence_seven_axis"]
        if not (isinstance(s, list) and len(s) == 7
                and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                        and 0.0 <= x <= 1.0 for x in s)):
            errors.append("confidence_seven_axis must be 7 numbers in [0,1]")

    # 5. optional independent hash cross-check (string equality only)
    hash_match = None
    if expected_graph_hash is not None:
        hash_match = (gh == expected_graph_hash)
        if not hash_match:
            errors.append("graph_hash does not match expected (tamper / wrong graph)")

    return {"valid": not errors, "errors": errors,
            "checks": {"schema_version": SCHEMA_VERSION, "method": method,
                       "hash_cross_checked": expected_graph_hash is not None,
                       "hash_match": hash_match}}


def verify_json(text: str, expected_graph_hash: str | None = None) -> dict:
    """Verify a credential given as a JSON string (the on-the-wire form)."""
    try:
        cred = json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        return {"valid": False, "errors": [f"invalid JSON: {e}"], "checks": {}}
    return verify_credential(cred, expected_graph_hash=expected_graph_hash)
