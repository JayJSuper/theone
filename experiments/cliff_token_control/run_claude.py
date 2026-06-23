"""Fourth-family completion of the token-length confound control: claude-opus-4-8
(cliff at k7, the latest-collapsing / strongest base). Same decoupling: isolated
distractor roots add prompt tokens but not 2^k marginalization load. If a LONG
low-load prompt stays accurate while a SHORTER high-load prompt collapses, length
is controlled (even inverted) and the cliff is 2^k load — now across four families.

  source ~/.theone_keys.env && .venv/bin/python experiments/cliff_token_control/run_claude.py
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
MODEL = "claude-opus-4-8"


def ask_claude(text):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps({"model": MODEL, "max_tokens": 4096,
                             "system": R.SYS,
                             "messages": [{"role": "user", "content": text + R.PROTO}]}).encode(),
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = "".join(b.get("text", "") for b in out.get("content", []))
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "fail": str(e)[:90]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1),
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    # claude collapse end is k7 (2^7=128); pad k2 past k7's prompt length
    cells = [("len k2 d0", 2, 0), ("len k2 d200", 2, 200),
             ("len k2 d440 (>k7 len, low)", 2, 440),
             ("cliff k6 d0", 6, 0), ("cliff k7 d0 (short, high)", 7, 0)]
    jpath = HERE / "rows_claude.jsonl"
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
            g = M.k_graph_padded(k, d, 74000 + 1000 * k + 7 * d + i)
            truth = round(InterventionEngine(M.truth_graph(g, k)).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            text = R.render(g, k)
            cl = ask_claude(text)
            row = {"label": label, "k": k, "d": d, "i": i, "truth": truth,
                   "prompt_words": len(text.split()), "claude": cl}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[{label:28} #{i:02d}] words={row['prompt_words']:>4} "
                  f"truth={truth:.3f} cl={cl.get('pred')} ({cl.get('latency')}s)", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== claude-opus-4-8 confound control: acc vs 2^k and prompt length ===")
    print(f"{'cell':>28} | {'2^k':>4} | {'words':>5} | {'acc(±.005)':>10} | proto-fail")
    summ = {}
    for label, k, d in cells:
        kr = [r for r in rows if r["label"] == label]
        if not kr:
            continue
        gp = [r for r in kr if r["claude"]["pred"] is not None]
        hit = sum(abs(r["claude"]["pred"] - r["truth"]) <= TOL for r in gp)
        words = int(np.mean([r["prompt_words"] for r in kr]))
        nfail = len(kr) - len(gp)
        summ[label] = {"two_k": 2 ** k, "words": words,
                       "acc": round(hit / len(kr), 3), "protocol_fails": nfail}
        print(f"{label:>28} | {2**k:>4} | {words:>5} | {hit/len(kr):>10.2f} | {nfail}/{len(kr)}")
    (HERE / "results_claude.json").write_text(json.dumps(summ, indent=2))
    print("\nKill shot: 'k2 d300 (>k7 len, low)' vs 'k7 d0 (short, high)' — LONGER low-load "
          "accurate while shorter high-load collapses => 2^k load, not length (4th family).")


if __name__ == "__main__":
    main()
