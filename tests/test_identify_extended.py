"""Front-door + IV identification tests (Q-C14). Textbook graphs; verifies the
priority order and the unverifiable-assumption flags."""
import pytest
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.identify import (find_frontdoor_set, find_instrument,
                                     identify_effect, check_frontdoor,
                                     check_instrument)


def _g(nodes, edges):
    g = CausalGraph()
    for n in nodes:
        g.add_variable(Variable(n))
    for a, b in edges:
        g.add_edge(a, b)
    return g


def frontdoor_graph():
    """Smoking(X) -> Tar(M) -> Cancer(Y), with UNOBSERVED genotype U->X, U->Y."""
    return _g(("U", "X", "M", "Y"),
              [("U", "X"), ("X", "M"), ("M", "Y"), ("U", "Y")])


def iv_graph():
    """Z -> X -> Y, with unobserved U -> X, U -> Y. Z is a valid instrument."""
    return _g(("Z", "U", "X", "Y"),
              [("Z", "X"), ("U", "X"), ("X", "Y"), ("U", "Y")])


class TestFrontDoor:
    def test_frontdoor_detected_when_confounder_unobserved(self):
        g = frontdoor_graph()
        # U unobserved -> no backdoor set; front-door via M
        res = identify_effect(g, "X", "Y", observed={"X", "M", "Y"})
        assert res["identifiable"] and res["strategy"] == "front_door"
        assert res["mediator_set"] == ["M"]
        assert any("not verifiable" in a.lower() for a in res["assumptions"])

    def test_frontdoor_criterion_direct(self):
        assert check_frontdoor(frontdoor_graph(), "X", "Y", {"M"}) is True

    def test_no_frontdoor_when_mediator_leaks(self):
        # add U->M : now backdoor X->M is open, front-door fails
        g = frontdoor_graph(); g.add_edge("U", "M")
        assert check_frontdoor(g, "X", "Y", {"M"}) is False


class TestInstrument:
    def test_iv_detected_when_confounder_unobserved(self):
        g = iv_graph()
        res = identify_effect(g, "X", "Y", observed={"Z", "X", "Y"})
        assert res["identifiable"] and res["strategy"] == "instrumental_variable"
        assert res["instrument"] == "Z"
        assert any("exclusion" in a.lower() for a in res["assumptions"])

    def test_invalid_instrument_with_direct_path_to_Y(self):
        # Z -> Y directly violates exclusion restriction
        g = iv_graph(); g.add_edge("Z", "Y")
        assert check_instrument(g, "X", "Y", "Z") is False


class TestPriorityAndRefusal:
    def test_backdoor_preferred_when_confounder_observed(self):
        g = frontdoor_graph()
        res = identify_effect(g, "X", "Y", observed={"U", "X", "M", "Y"})
        assert res["strategy"] == "backdoor" and res["adjustment_set"] == ["U"]

    def test_refuse_when_nothing_identifies(self):
        # X<-U->Y, U unobserved, no mediator, no instrument
        g = _g(("U", "X", "Y"), [("U", "X"), ("U", "Y"), ("X", "Y")])
        res = identify_effect(g, "X", "Y", observed={"X", "Y"})
        assert res["identifiable"] is False and res["strategy"] is None
