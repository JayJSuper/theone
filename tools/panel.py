"""panel.py — the lean kernel's first half: MULTI-VOICE.

The valuable core of the "agent cluster" idea, with zero infrastructure: ask one
question to several frontier models AT ONCE and put their *disagreement* in front
of you. The value is not "division of labour" — it is that when DeepSeek says yes
and Claude says no, the disagreement itself is information.

No Ray, no Docker, no queue. Just urllib + threads. Each voice reuses the exact
call pattern already proven in experiments/. Keys come from ~/.theone_keys.env
(`source ~/.theone_keys.env` first); a missing key simply drops that voice.

CLI:   source ~/.theone_keys.env && python tools/panel.py "your question"
       python tools/panel.py --voices gpt5,claude,deepseek "..."
API:   from tools.panel import panel; r = panel("..."); print(r["disagreement"])
"""
from __future__ import annotations
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

TIMEOUT = 240  # reasoning models (gpt-5.1) are slow; generous per-voice ceiling


def _post(url: str, payload: dict, headers: dict, timeout: int = TIMEOUT) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# --- the four voices: each returns plain text, or raises ------------------------
def _gpt5(q: str) -> str:
    out = _post(
        "https://api.openai.com/v1/chat/completions",
        {"model": "gpt-5.1", "messages": [{"role": "user", "content": q}],
         "max_completion_tokens": 4096},
        {"Content-Type": "application/json",
         "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    return out["choices"][0]["message"]["content"].strip()


def _claude(q: str) -> str:
    out = _post(
        "https://api.anthropic.com/v1/messages",
        {"model": "claude-opus-4-8", "max_tokens": 4096,
         "messages": [{"role": "user", "content": q}]},
        {"x-api-key": os.environ["ANTHROPIC_API_KEY"],
         "anthropic-version": "2023-06-01", "content-type": "application/json"})
    return "".join(b.get("text", "") for b in out.get("content", [])).strip()


def _deepseek(q: str) -> str:
    out = _post(
        "https://api.deepseek.com/chat/completions",
        {"model": "deepseek-v4-flash",
         "messages": [{"role": "user", "content": q}],
         "max_tokens": 4096, "temperature": 0.3},
        {"Content-Type": "application/json",
         "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
    return out["choices"][0]["message"]["content"].strip()


def _gemini(q: str) -> str:
    key = os.environ["GEMINI_API_KEY"]
    out = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-pro:generateContent?key={key}",
        {"contents": [{"parts": [{"text": q}]}],
         "generationConfig": {"temperature": 0.3, "maxOutputTokens": 24000}},
        {"Content-Type": "application/json"})
    parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()


VOICES = {"gpt5": _gpt5, "claude": _claude, "deepseek": _deepseek, "gemini": _gemini}
ENVKEY = {"gpt5": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY",
          "deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}


def _one(name: str, q: str) -> dict:
    if not os.environ.get(ENVKEY[name]):
        return {"voice": name, "ok": False, "error": f"{ENVKEY[name]} not set"}
    t0 = time.time()
    try:
        text = VOICES[name](q)
        return {"voice": name, "ok": True, "text": text,
                "latency_s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001 — a dead voice must not kill the panel
        return {"voice": name, "ok": False,
                "error": str(e)[:160], "latency_s": round(time.time() - t0, 1)}


def panel(question: str, voices: list[str] | None = None) -> dict:
    """Ask all voices concurrently. Returns {question, answers[], disagreement}.

    `disagreement` is a one-line human cue, not a model judgement — surfacing the
    split is the product; resolving it is your job (and, when it matters, a human
    expert's, not another model's)."""
    names = voices or list(VOICES)
    with cf.ThreadPoolExecutor(max_workers=len(names)) as ex:
        answers = list(ex.map(lambda n: _one(n, question), names))
    live = [a for a in answers if a["ok"]]
    note = (f"{len(live)}/{len(names)} voices answered; "
            "read them side by side — divergence is the signal." if live
            else "no voice answered (check keys / network).")
    return {"question": question, "answers": answers, "disagreement": note}


def _print(r: dict) -> None:
    print(f"\n=== PANEL: {r['question']}\n{r['disagreement']}\n")
    for a in r["answers"]:
        head = f"── {a['voice']}  ({a.get('latency_s','?')}s)"
        if a["ok"]:
            print(f"{head}\n{a['text']}\n")
        else:
            print(f"{head}  [SKIPPED: {a['error']}]\n")


if __name__ == "__main__":
    argv = sys.argv[1:]
    sel = None
    if argv and argv[0] == "--voices":
        sel = argv[1].split(","); argv = argv[2:]
    if not argv:
        print("usage: python tools/panel.py [--voices gpt5,claude,...] \"question\"")
        sys.exit(1)
    _print(panel(" ".join(argv), sel))
