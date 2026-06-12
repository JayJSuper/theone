"""The One local hybrid server — serves the universal demo AND a real
S1+S2 pipeline at /api/chat (same origin, no CORS, key never reaches the browser).

Run:  source ~/.theone_keys.env && python -m theone.server [--port 8779]
S1 (DeepSeek) is optional: without a key the pipeline still runs — S2 numbers
render through the deterministic template and S1-direct questions get an honest
"S1 offline" notice. Honesty surfaces in the product.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from . import __version__
from .hybrid import (build_library, route, s2_answer, s1_render,
                     template_render, identify_gate, answer_causal)
from .memory.store import MemoryStore

DEMO_DIR = Path(__file__).resolve().parents[2] / "demo" / "one_universal"
MEM_PATH = str(Path.home() / ".theone_demo_memory.db")

_LIB = build_library()
_S1 = None
if os.environ.get("DEEPSEEK_API_KEY"):
    from .llm import DeepSeekClient
    _S1 = DeepSeekClient()


def handle_chat(query: str) -> dict:
    t0 = time.time()
    r = route(query, _LIB)
    mode = r["mode"]
    base = {"mode": mode, "engine_version": __version__}

    if mode == "memory_put":
        mem = MemoryStore(MEM_PATH)
        mid = mem.put("note", r["payload"], source="user_chat")
        mem.close()
        return {**base, "answer": f"Saved to your local memory (#{mid}). "
                "It never leaves this machine.",
                "receipt": {"method": "memory_put", "memory_id": mid}}

    if mode == "memory_get":
        mem = MemoryStore(MEM_PATH)
        hits = mem.search("note")
        mem.close()
        ans = ("From your local memory: " +
               "; ".join(f"#{h['id']} {h['value']}" for h in hits[:3])
               ) if hits else "Nothing in local memory yet. Say 'remember ...' first."
        return {**base, "answer": ans,
                "receipt": {"method": "memory_search", "hits": len(hits)}}

    if mode == "s2_causal":
        gate = identify_gate(r["graph"])
        if not gate["identifiable"]:
            miss = ", ".join(gate["missing"])
            return {**base, "mode": "abstain_unidentifiable",
                    "answer": ("I can't answer this causally — and I can prove "
                               f"why: the required adjustment variable ({miss}) was "
                               "never observed, so the causal effect is "
                               "unidentifiable from available data. Measuring "
                               f"{miss} is exactly what would change that."),
                    "receipt": {"method": "identifiability_gate_refusal",
                                "graph_hash": r["graph"].graph.content_hash(),
                                "missing_variables": gate["missing"],
                                "limits": gate["reason"], "confidence": 0.95}}
        s2 = answer_causal(r["graph"], gate)
        answer = s1_render(query, s2, _S1)
        limits = s2["note"]
        if s2.get("assumptions"):
            limits += " | " + " ; ".join(s2["assumptions"])
        return {**base, "answer": answer, "s1_used": _S1 is not None,
                "receipt": {
                    "method": s2["method"], "graph_hash": s2["graph_hash"],
                    "strategy": s2["strategy"],
                    "adjustment_set": s2["adjustment_set"],
                    "mediator": s2.get("mediator"),
                    "numbers": {k: s2[k] for k in
                                ("obs_ate", "int_ate", "confounding_bias",
                                 "int_do_x1", "int_do_x0")},
                    "evidence_tier": s2["evidence_tier"], "limits": limits,
                    "confidence": 0.86 if s2["strategy"] == "backdoor" else 0.72},
                "latency_s": round(time.time() - t0, 3)}

    if mode in ("abstain_no_model", "abstain_forecast"):
        why = ("no validated causal model covers this domain yet — answering "
               "would be fabrication" if mode == "abstain_no_model" else
               "long-horizon market prices have no identifiable causal path — "
               "any number would be fabrication")
        return {**base, "answer": "I don't know — and I won't guess: " + why +
                ". I can describe what evidence would change that.",
                "receipt": {"method": "calibrated_abstention", "confidence": 0.18,
                            "limits": "refusal is itself the calibrated output"}}

    # s1_direct
    if _S1 is None:
        return {**base, "answer": "S1 (language organ) is offline — no API key "
                "loaded. Causal and memory functions still work. "
                "(Honest notice, not an error.)", "s1_used": False}
    out = _S1.chat([{"role": "system", "content":
                     "You are The One's S1 organ. Be helpful, concise, honest; "
                     "answer in the user's language. If asked for facts you are "
                     "unsure of, say so plainly."},
                    {"role": "user", "content": query}], max_tokens=600)
    return {**base, "answer": out["content"].strip(), "s1_used": True,
            "receipt": {"method": "s1_direct_unverified",
                        "limits": "statistical answer from the S1 organ; not "
                        "causally verified — no do-query was run",
                        "model": out.get("model")},
            "latency_s": round(time.time() - t0, 3)}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DEMO_DIR), **kw)

    def log_message(self, *a):                       # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._json({"ok": True, "version": __version__,
                               "s1": _S1 is not None,
                               "graphs": [g.key for g in _LIB]})
        return super().do_GET()

    def do_POST(self):
        if self.path.startswith("/api/chat"):
            n = int(self.headers.get("Content-Length", 0))
            try:
                q = json.loads(self.rfile.read(n)).get("query", "").strip()
                if not q:
                    raise ValueError("empty query")
                return self._json(handle_chat(q))
            except Exception as e:
                return self._json({"error": str(e)[:200]}, code=400)
        self.send_error(404)

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8779)
    args = ap.parse_args()
    print(f"The One hybrid server v{__version__}  http://localhost:{args.port}"
          f"  S1={'on' if _S1 else 'off'}  graphs={[g.key for g in _LIB]}")
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
