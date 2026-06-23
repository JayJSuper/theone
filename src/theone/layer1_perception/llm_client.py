"""L1 · live LLM client — calls a mounted external LLM as a perception/generation organ.

The One does NOT rebuild the language model; it mounts one. This client speaks to
DeepSeek (OpenAI-compatible) or Anthropic over plain stdlib HTTP. It loads API keys from
~/.theone_keys.env at runtime and NEVER logs, prints, or persists them. If no key is
configured or the network is unavailable, it degrades gracefully to a clearly-labelled
offline stub, so the product runs end-to-end either way (the verifiable kernel needs no
API; only the chat/code generation path uses this).
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def load_keys(path: str = "~/.theone_keys.env") -> None:
    """Parse the keys file into os.environ (idempotent; values never echoed)."""
    p = Path(path).expanduser()
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_PROVIDERS = {
    "deepseek": {"url": "https://api.deepseek.com/chat/completions",
                 "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat", "kind": "openai"},
    "openai":   {"url": "https://api.openai.com/v1/chat/completions",
                 "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini", "kind": "openai"},
    "claude":   {"url": "https://api.anthropic.com/v1/messages",
                 "key_env": "ANTHROPIC_API_KEY", "model": "claude-3-5-sonnet-20241022", "kind": "anthropic"},
}


@dataclass
class LLMReply:
    text: str
    provider: str
    live: bool          # True = real API call; False = offline stub
    note: str = ""


class LLMClient:
    def __init__(self, provider: str = "deepseek", timeout: float = 30.0) -> None:
        load_keys()
        if provider not in _PROVIDERS:
            raise ValueError(f"unknown provider {provider}; choose {list(_PROVIDERS)}")
        self.provider = provider
        self.cfg = _PROVIDERS[provider]
        self.timeout = timeout

    def available(self) -> bool:
        return bool(os.environ.get(self.cfg["key_env"]))

    def _request(self, prompt: str, system: Optional[str]) -> str:
        key = os.environ[self.cfg["key_env"]]
        if self.cfg["kind"] == "openai":
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            body = {"model": self.cfg["model"], "messages": msgs, "temperature": 0.3}
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        else:  # anthropic
            body = {"model": self.cfg["model"], "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}]}
            if system:
                body["system"] = system
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01",
                       "Content-Type": "application/json"}
        req = urllib.request.Request(self.cfg["url"], data=json.dumps(body).encode(),
                                     headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        if self.cfg["kind"] == "openai":
            return data["choices"][0]["message"]["content"]
        return data["content"][0]["text"]

    def chat(self, prompt: str, system: Optional[str] = None) -> LLMReply:
        if not self.available():
            return LLMReply(
                text=f"[offline stub · {self.provider} key not configured] echo: {prompt[:120]}",
                provider=self.provider, live=False, note="no API key")
        try:
            return LLMReply(self._request(prompt, system), self.provider, live=True)
        except (urllib.error.URLError, KeyError, TimeoutError, json.JSONDecodeError) as e:
            return LLMReply(
                text=f"[offline stub · {self.provider} unreachable] echo: {prompt[:120]}",
                provider=self.provider, live=False, note=f"{type(e).__name__}")


__all__ = ["LLMClient", "LLMReply", "load_keys"]
