"""Cross-family replication of the token-length confound control on deepseek-v4-flash
(different family from gpt-5.1; its cliff is at k4, the earliest). Same decoupling:
isolated distractor roots add prompt tokens but not marginalization load. If a LONG
low-load prompt stays accurate while a SHORTER high-load prompt collapses, the cliff
is 2^k load, not token length — replicated outside the gpt family.

  source ~/.theone_keys.env && python experiments/cliff_token_control/run_deepseek.py
"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_m = importlib.util.spec_from_file_location("ctc", HERE / "run.py")
M = importlib.util.module_from_spec(_m); _m.loader.exec_module(M)
R = M.R                      # complexity_axis/run.py (render + ask_deepseek)
N = 12
TOL = 0.005


def main():
    # deepseek cliff is k4, so the collapse end uses k4 (not k5)
    cells = [("len k2 d0", 2, 0), ("len k2 d45", 2, 45),
             ("len k2 d120 (longest, low)", 2, 120),
             ("cliff k3 d0", 3, 0), ("cliff k4 d0 (short, high)", 4, 0)]
    jpath = HERE / "rows_deepseek.jsonl"
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
            g = M.k_graph_padded(k, d, 52000 + 1000 * k + 7 * d + i)
            truth = round(InterventionEngine(M.truth_graph(g, k)).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            text = R.render(g, k)
            ds = R.ask_deepseek(text)
            row = {"label": label, "k": k, "d": d, "i": i, "truth": truth,
                   "prompt_words": len(text.split()), "deepseek": ds}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[{label:28} #{i:02d}] words={row['prompt_words']:>4} "
                  f"truth={truth:.3f} ds={ds.get('pred')}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== deepseek confound control: acc vs 2^k and prompt length ===")
    print(f"{'cell':>28} | {'2^k':>4} | {'words':>5} | {'ds acc(±.005)':>13}")
    summ = {}
    for label, k, d in cells:
        kr = [r for r in rows if r["label"] == label]
        gp = [r for r in kr if r["deepseek"]["pred"] is not None]
        # protocol failure (no parseable ANSWER) counts as error, per AM-007
        acc = np.mean([abs(r["deepseek"]["pred"] - r["truth"]) <= TOL for r in gp]) \
            if gp else 0.0
        acc = acc * (len(gp) / len(kr)) if kr else None   # fold protocol fails into denom
        words = int(np.mean([r["prompt_words"] for r in kr])) if kr else 0
        nfail = sum(1 for r in kr if r["deepseek"]["pred"] is None)
        summ[label] = {"two_k": 2 ** k, "words": words,
                       "acc": round(acc, 3) if acc is not None else None,
                       "protocol_fails": nfail}
        print(f"{label:>28} | {2**k:>4} | {words:>5} | "
              f"{str(summ[label]['acc']):>13}  (proto-fail {nfail}/{len(kr)})")
    (HERE / "results_deepseek.json").write_text(json.dumps(summ, indent=2))
    print("\nKill shot: 'k2 d120 (longest,low)' vs 'k4 d0 (short,high)' — if the LONGER "
          "low-load prompt stays accurate and the shorter high-load one collapses,\n"
          "the cliff is 2^k load, not token length — replicated on deepseek.")


if __name__ == "__main__":
    main()
