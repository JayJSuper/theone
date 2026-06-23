"""Unit tests for the credential spine — the cross-cutting invariant of the fusion."""
from __future__ import annotations
import pytest

from theone.core.spine import (
    Decision, Credential, LayerVerdict, CredentialedLayer, Spine,
)
from theone.core.exceptions import ContractViolationError


class _Good(CredentialedLayer):
    name, layer_index = "good", 0

    def process(self, inputs):
        c = Credential("good", "x=1", value=1.0, recompute=lambda: 1.0, tolerance=1e-9)
        return LayerVerdict.answer(self.name, c, value=inputs)


class _Liar(CredentialedLayer):
    """Claims a value its recompute does not reproduce."""
    name, layer_index = "liar", 1

    def process(self, inputs):
        c = Credential("liar", "x=1", value=1.0, recompute=lambda: 2.0, tolerance=1e-9)
        return LayerVerdict.answer(self.name, c, value=inputs)


class _NoCred(CredentialedLayer):
    name, layer_index = "nocred", 0

    def process(self, inputs):
        return LayerVerdict(Decision.ANSWER, self.name, credential=None)


class _BadAbstain(CredentialedLayer):
    name, layer_index = "badabstain", 0

    def process(self, inputs):
        return LayerVerdict(Decision.ABSTAIN, self.name, reason=None)


def test_good_layer_answers_and_verifies():
    v = _Good().run({"a": 1})
    assert v.is_answer()
    ok, info = v.credential.verify()
    assert ok and info["gap"] == 0.0


def test_failing_recompute_is_downgraded_to_abstain():
    v = _Liar().run({})
    assert not v.is_answer()
    assert "did not recompute" in v.reason


def test_answer_without_credential_raises():
    with pytest.raises(ContractViolationError):
        _NoCred().run({})


def test_abstain_without_reason_raises():
    with pytest.raises(ContractViolationError):
        _BadAbstain().run({})


def test_spine_all_answer_stacks_credentials():
    sv = Spine([_Good()]).run({"a": 1})
    assert sv.is_answer() and len(sv.credentials) == 1


def test_spine_abstains_at_failing_layer_and_short_circuits():
    # _Good (idx 0) answers, _Liar (idx 1) is downgraded to abstain -> system abstains there
    sv = Spine([_Good(), _Liar()]).run({"a": 1})
    assert not sv.is_answer()
    assert sv.abstained_at == "liar"


class _ThrowingRecompute(CredentialedLayer):
    """A credential whose recompute raises — must fail safe, not crash the spine."""
    name, layer_index = "throws", 0

    def process(self, inputs):
        def _boom():
            raise ValueError("recompute blew up")
        c = Credential("throws", "x=1", value=1.0, recompute=_boom, tolerance=1e-9)
        return LayerVerdict.answer(self.name, c, value=inputs)


def test_recompute_exception_fails_safe_to_abstain():
    v = _ThrowingRecompute().run({})
    assert not v.is_answer()          # did not crash; downgraded to abstain
    assert "did not recompute" in v.reason
    sv = Spine([_ThrowingRecompute()]).run({})
    assert not sv.is_answer() and sv.abstained_at == "throws"


def test_string_and_bool_credentials_verify():
    cs = Credential("s", "k", value="abc", recompute=lambda: "abc")
    cb = Credential("b", "k", value=True, recompute=lambda: True)
    assert cs.verify()[0] and cb.verify()[0]
    cbad = Credential("s", "k", value="abc", recompute=lambda: "xyz")
    assert not cbad.verify()[0]
