"""Clean cross-family token-length control on gemini-2.5-pro (different family from
gpt-5.1; protocol-stable, cliff at k6, 24k output budget). deepseek was protocol-
fragile under long prompts and thus a noisy subject for the length decoupling; gemini
is the clean second subject. Collapse end uses k6 (2^6=64).

Kill shot: k2 d240 (306 words, LONGER than k6's 282) at load 4 vs k6 d0 (282 words)
at load 64. If the LONGER low-load prompt stays accurate and the shorter high-load one
collapses, length is controlled (even inverted) and the cliff is 2^k load — replicated
clean on a second family.

  source ~/.theone_keys.env && python experiments/cliff_token_control/run_gemini.py
"""
from __future__ import annotations
import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_m = importlib.util.spec_from_file_location("ctc", HERE / "run.py")
M = importlib.util.module_from_spec(_m); _m.loader.exec_module(M)
R = M.R
N = 8
TOL = 0.005


def ask_gemini(text, maxtok=24000):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-pro:generateContent?key={os.environ['GEMINI_API_KEY']}",
            data=json.dumps({"systemInstruction": {"parts": [{"text": R.SYS}]},
                             "contents": [{"parts": [{"text": text + R.PROTO}]}],
                             "generationConfig": {"temperature": 0,
                                                  "maxOutputTokens": maxtok}}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        c = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "fail": str(e)[:90]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1),
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    cells = [("len k2 d0", 2, 0), ("len k2 d120", 2, 120),
             ("len k2 d240 (>k6 len, low)", 2, 240),
             ("cliff k5 d0", 5, 0), ("cliff k6 d0 (short, high)", 6, 0)]
    jpath = HERE / "rows_gemini.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["label"], r["i"]))
    jf = jpath.open("a")
    for label, k, d in cells:
        for i in range(N):
            if (label, i) in done:
                continue
            g = M.k_graph_padded(k, d, 63000 + 1000 * k + 7 * d + i)
            truth = round(InterventionEngine(M.truth_graph(g, k)).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            text = R.render(g, k)
            gm = ask_gemini(text)
            row = {"label": label, "k": k, "d": d, "i": i, "truth": truth,
                   "prompt_words": len(text.split()), "gemini": gm}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[{label:28} #{i:02d}] words={row['prompt_words']:>4} "
                  f"truth={truth:.3f} gm={gm.get('pred')} ({gm.get('latency')}s)", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== gemini-2.5-pro confound control: acc vs 2^k and prompt length ===")
    print(f"{'cell':>28} | {'2^k':>4} | {'words':>5} | {'acc(±.005)':>10} | proto-fail")
    summ = {}
    for label, k, d in cells:
        kr = [r for r in rows if r["label"] == label]
        if not kr:
            continue
        gp = [r for r in kr if r["gemini"]["pred"] is not None]
        hit = sum(abs(r["gemini"]["pred"] - r["truth"]) <= TOL for r in gp)
        words = int(np.mean([r["prompt_words"] for r in kr]))
        nfail = len(kr) - len(gp)
        summ[label] = {"two_k": 2 ** k, "words": words,
                       "acc": round(hit / len(kr), 3), "protocol_fails": nfail}
        print(f"{label:>28} | {2**k:>4} | {words:>5} | {hit/len(kr):>10.2f} | "
              f"{nfail}/{len(kr)}")
    (HERE / "results_gemini.json").write_text(json.dumps(summ, indent=2))
    print("\nKill shot: 'k2 d240 (>k6 len, low)' vs 'k6 d0 (short, high)' — LONGER low-load "
          "prompt accurate while shorter high-load collapses => 2^k load, not length.")


if __name__ == "__main__":
    main()
