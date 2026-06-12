"""Credential v1 + independent verifier tests (T2-08).
Verifies (a) real orchestrator credentials pass, (b) tampered ones are caught by
the independent verifier, (c) the verifier shares no code with the generator."""
import re
import time
from pathlib import Path
import pytest
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.memory.store import MemoryStore
from theone.agent.orchestrator import Orchestrator
from theone.credential.verifier import verify_credential, verify_json
from theone.credential.schema import CREDENTIAL_SCHEMA_V1, ALLOWED_METHODS


def _confounded():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("X", {(1,): {1: 0.8, 0: 0.2}, (0,): {1: 0.2, 0: 0.8}})
    g.set_cpt("Y", {(1, 1): {1: 0.9, 0: 0.1}, (0, 1): {1: 0.5, 0: 0.5},
                    (1, 0): {1: 0.6, 0: 0.4}, (0, 0): {1: 0.2, 0: 0.8}})
    return g


@pytest.fixture
def agent(tmp_path):
    eng = InterventionEngine(_confounded())
    return Orchestrator(eng, MemoryStore(str(tmp_path / "m.db"))), eng


class TestCredentialV1:
    def test_real_credentials_all_methods_pass(self, agent):
        ag, eng = agent
        for req in ("P(Y=1|do(X=1))", "P(Y=1|X=1)", "remember beachhead=medical",
                    "recall beachhead", "write me a poem"):
            cred = ag.handle(req).credential
            v = verify_credential(cred, expected_graph_hash=eng.g.content_hash())
            assert v["valid"], (req, v["errors"])
            assert v["checks"]["hash_match"] is True

    def test_method_enum_matches_orchestrator_outputs(self, agent):
        ag, _ = agent
        cred = ag.handle("P(Y=1|do(X=1))").credential
        assert cred["method"] in ALLOWED_METHODS

    def test_tamper_is_caught(self, agent):
        ag, eng = agent
        good = ag.handle("P(Y=1|do(X=1))").credential
        # bad method
        bad = dict(good, method="totally_made_up")
        assert verify_credential(bad)["valid"] is False
        # malformed hash
        assert verify_credential(dict(good, graph_hash="xyz"))["valid"] is False
        # missing required field
        miss = dict(good); miss.pop("timestamp")
        assert verify_credential(miss)["valid"] is False
        # do-method missing adjustment_set
        miss2 = dict(good); miss2.pop("adjustment_set")
        assert verify_credential(miss2)["valid"] is False
        # hash mismatch vs expected
        assert verify_credential(good, expected_graph_hash="0" * 64)["valid"] is False

    def test_optional_field_range_checks(self):
        base = {"query": "q", "method": "unrouted", "graph_hash": "a" * 64,
                "timestamp": 1.0, "engine_version": "0.1.0"}
        assert verify_credential(base)["valid"] is True
        assert verify_credential(dict(base, value=1.5))["valid"] is False      # >1
        assert verify_credential(dict(base, value=0.5))["valid"] is True
        assert verify_credential(dict(base,
                 confidence_seven_axis=[0.1] * 6))["valid"] is False           # not 7
        assert verify_credential(dict(base,
                 confidence_seven_axis=[0.1] * 7))["valid"] is True
        assert verify_credential(dict(base, memory_id=-1))["valid"] is False

    def test_verify_json_roundtrip(self, agent):
        import json
        ag, eng = agent
        cred = ag.handle("P(Y=1|X=1)").credential
        v = verify_json(json.dumps(cred), expected_graph_hash=eng.g.content_hash())
        assert v["valid"] is True
        assert verify_json("{not json")["valid"] is False

    def test_separation_invariant_verifier_imports_no_generator(self):
        """Charter: zero shared logic. Parse the verifier's actual import statements
        (not docstrings) and assert none reach the generator stack."""
        import ast
        src = Path(verify_credential.__globals__["__file__"]).read_text()
        imported = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        joined = " ".join(imported)
        for forbidden in ("orchestrator", "agent", "causal", "engine", "bench"):
            assert forbidden not in joined, f"verifier imports generator stack: {imported}"

    def test_schema_is_wellformed(self):
        assert CREDENTIAL_SCHEMA_V1["title"].startswith("The One")
        assert CREDENTIAL_SCHEMA_V1["properties"]["graph_hash"]["pattern"] == r"^[0-9a-f]{64}$"
