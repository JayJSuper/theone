"""DeepSeek S1 backend tests.
Unit tests run anywhere (no network). The integration test runs ONLY when
DEEPSEEK_API_KEY is present (skipped in CI - key never enters the repo/CI)."""
import os
import pytest
from theone.llm import DeepSeekClient, LLMError


class TestDeepSeekUnit:
    def test_missing_key_raises_clean_error(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(LLMError, match="DEEPSEEK_API_KEY not set"):
            DeepSeekClient().chat([{"role": "user", "content": "hi"}])

    def test_key_never_stored_on_instance(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        c = DeepSeekClient()
        leaked = [v for v in vars(c).values() if isinstance(v, str) and "test-key" in v]
        assert leaked == []          # key must not persist on the instance
        assert "test-key" not in repr(vars(c))

    def test_defaults(self):
        c = DeepSeekClient()
        assert c.model == "deepseek-v4-flash"
        assert c.max_retries >= 1


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"),
                    reason="real-API integration: requires DEEPSEEK_API_KEY (E-01)")
class TestDeepSeekIntegrationReal:
    def test_real_chat_roundtrip(self):
        # deepseek-v4-flash is a reasoning model: thinking tokens are billed from
        # max_tokens BEFORE the visible answer (observed live: 33 reasoning tokens
        # on a trivial prompt). Budget must cover thinking + answer.
        out = DeepSeekClient().chat(
            [{"role": "user", "content": "Reply with exactly: PONG"}],
            max_tokens=128, temperature=0.0)
        assert "PONG" in out["content"].upper()
        assert out["usage"]["total_tokens"] > 0
        assert out["latency_s"] > 0
