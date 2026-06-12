"""DeepSeek backend — The One's first real S1 (fast/statistical) organ.

M1 step 1 of the master blueprint: S1 handles understanding / translation /
expression; it must NEVER be asked to do causal math (that is the deterministic
S2 engine's job — the L1-parrot rule from the architecture review).

Security invariants:
- API key is read from the DEEPSEEK_API_KEY environment variable at call time;
  it is never stored on the instance, never logged, never serialized.
- stdlib urllib only (no new runtime dependencies).

Evidence: first real call archived at experiments/e01_deepseek_real/ (E-01).
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error

API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"


class LLMError(RuntimeError):
    """Raised on missing key, transport failure, or malformed response."""


class DeepSeekClient:
    """Minimal, dependency-free DeepSeek chat client.

    >>> s1 = DeepSeekClient()
    >>> out = s1.chat([{"role": "user", "content": "hi"}])
    >>> out["content"], out["usage"]["total_tokens"]
    """

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 60.0,
                 max_retries: int = 2) -> None:
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    @staticmethod
    def _key() -> str:
        key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise LLMError(
                "DEEPSEEK_API_KEY not set. Load it in your shell, e.g. "
                "`source ~/.theone_keys.env` (keys live outside the repo).")
        return key

    def chat(self, messages: list, *, max_tokens: int = 512,
             temperature: float = 0.2, model: str | None = None) -> dict:
        """One chat completion. Returns {content, usage, model, id, latency_s}.
        Raises LLMError on failure (after retries with backoff)."""
        payload = json.dumps({
            "model": model or self.model, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
        }).encode()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                API_URL, data=payload, method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self._key()}"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode())
                content = body["choices"][0]["message"]["content"]
                return {"content": content, "usage": body.get("usage", {}),
                        "model": body.get("model"), "id": body.get("id"),
                        "latency_s": round(time.time() - t0, 3)}
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                if e.code in (429, 500, 502, 503) and attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    last_err = LLMError(f"HTTP {e.code}: {detail}")
                    continue
                raise LLMError(f"HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, TimeoutError, KeyError,
                    json.JSONDecodeError) as e:
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    last_err = e
                    continue
                raise LLMError(f"transport/parse failure: {e}") from None
        raise LLMError(f"retries exhausted: {last_err}")
