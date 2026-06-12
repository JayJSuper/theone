"""Hybrid pipeline v0 tests — router cost-sensitivity, S2 numbers, S1 fallback."""
import pytest
from theone.hybrid import (build_library, route, s2_answer, template_render,
                           s1_render, _match_graph)

LIB = build_library()


class TestRouterV0:
    def test_causal_with_validated_graph_goes_s2(self):
        for q in ("咖啡会导致心脏病吗？", "Does coffee cause heart disease?",
                  "广告投放对销量的因果影响是什么"):
            r = route(q, LIB)
            assert r["mode"] == "s2_causal", q
            assert r["graph"].usable_for_do()

    def test_causal_without_model_abstains_never_s1(self):
        """Cost-sensitive core: causal intent + no validated graph -> ABSTAIN.
        Falling through to S1 would resurrect hallucination."""
        r = route("熬夜会导致脱发吗？", LIB)
        assert r["mode"] == "abstain_no_model"

    def test_forecast_abstains(self):
        assert route("明年比特币会涨到多少？", LIB)["mode"] == "abstain_forecast"
        assert route("Where will bitcoin be next year?", LIB)["mode"] == "abstain_forecast"

    def test_memory_routes(self):
        assert route("记住 我每天五点站桩", LIB)["mode"] == "memory_put"
        assert route("我上次让你记住什么了", LIB)["mode"] == "memory_get"

    def test_plain_chat_goes_s1(self):
        assert route("把这句话翻译成英文：你好", LIB)["mode"] == "s1_direct"

    def test_graph_match_needs_two_variables(self):
        assert _match_graph("我爱喝咖啡", LIB) is None          # 1 var only
        assert _match_graph("咖啡和心脏的关系", LIB).key == "coffee_heart"


class TestS2Numbers:
    def test_coffee_graph_matches_frozen_t1_truths(self):
        """coffee_heart is the T1 structure renamed: numbers must equal the
        frozen truth table (0.82/0.70/0.28/0.40 etc., tol 1e-6)."""
        lg = next(g for g in LIB if g.key == "coffee_heart")
        s2 = s2_answer(lg)
        assert s2["obs_given_x1"] == pytest.approx(0.82, abs=1e-6)
        assert s2["int_do_x1"] == pytest.approx(0.70, abs=1e-6)
        assert s2["obs_ate"] == pytest.approx(0.54, abs=1e-6)
        assert s2["int_ate"] == pytest.approx(0.30, abs=1e-6)
        assert s2["confounding_bias"] == pytest.approx(0.24, abs=1e-6)
        assert s2["adjustment_set"] == ["smoking"]
        assert "ILLUSTRATIVE" in s2["note"]                  # honesty surfaces

    def test_only_machine_validated_graphs_serve_do(self):
        for g in LIB:
            assert g.usable_for_do()                          # v0 ships validated only


class TestS1Fallback:
    def test_template_render_offline(self):
        s2 = s2_answer(LIB[0])
        txt = template_render(s2)
        assert "0.70" in txt and "smoking" in txt and "Scope:" in txt

    def test_s1_render_degrades_gracefully(self):
        class Broken:
            def chat(self, *a, **k):
                raise RuntimeError("network down")
        s2 = s2_answer(LIB[0])
        assert "0.70" in s1_render("q", s2, Broken())          # falls to template
        assert "0.70" in s1_render("q", s2, None)
