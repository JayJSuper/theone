"""DECISIVE CONFOUND CONTROL (external-reviewer request): is the combinatorial cliff
driven by marginalization load (2^k), or merely by prompt/token length growing with k
(long-context / arithmetic-execution degradation)?

We DECOUPLE the two. A confounded SCM with k confounders requires marginalizing 2^k
configs, but its prompt also grows with k. So we PAD a LOW-k problem with d isolated,
irrelevant distractor variables (each adds a P(D)=... line and tokens, but does NOT
change the do-query or its 2^k marginalization). Two sweeps:

  (1) length control: fix k=2 (marginalization = 4), vary d in {0,10,25,45} so the
      PROMPT grows long. If accuracy stays high, length/context is NOT the cause.
  (2) the decisive cell: k=2 + heavy padding (LONG prompt, LOW load) vs k=5 + no
      padding (SHORTER prompt, HIGH load). If the long-low is easy and the short-high
      collapses, the cliff is marginalization load, not token length.

Subject: gpt-5.1 (clean cliff at k=5). Engine exact throughout. Run:
  source ~/.theone_keys.env && python experiments/cliff_token_control/run.py
"""
from __future__ import annotations
import importlib.util, itertools, json
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
N = 20
TOL = 0.005


def k_graph_padded(k, d, seed):
    """k confounders (->X,->Y) + X->Y + d ISOLATED distractor roots (length padding,
    causally irrelevant — do(X=1) and its 2^k marginalization are unchanged by d)."""
    rng = np.random.default_rng(seed); g = CausalGraph()
    Us = [f"U{i}" for i in range(k)]; Ds = [f"D{i}" for i in range(d)]
    for n in Us + ["X", "Y"] + Ds:
        g.add_variable(Variable(n))
    for u in Us:
        g.add_edge(u, "X"); g.add_edge(u, "Y")
    g.add_edge("X", "Y")
    for u in Us + Ds:
        p = round(float(rng.uniform(.1, .9)), 3); g.set_cpt(u, {(): {1: p, 0: round(1 - p, 3)}})
    for v in ("X", "Y"):
        order = list(g.parent_order(v)); rows = {}
        for c in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(.1, .9)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
        g.set_cpt(v, rows)
    return g


def truth_graph(g, k):
    """Pruned copy with only U0..U{k-1}, X, Y. The d distractor roots are ISOLATED
    (no path to X or Y), so P(Y|do(X)) is identical — but enumeration is 2^k, not
    2^(k+d). The engine enumerates the full joint, so without this pruning d=25
    means 2^25 configs (CPU blowup) for a truth that doesn't depend on them.
    Parent orders for X,Y are unchanged by dropping distractors, so CPT rows align."""
    tg = CausalGraph()
    Us = [f"U{i}" for i in range(k)]
    for n in Us + ["X", "Y"]:
        tg.add_variable(Variable(n))
    for u in Us:
        tg.add_edge(u, "X"); tg.add_edge(u, "Y")
    tg.add_edge("X", "Y")
    for n in Us + ["X", "Y"]:
        tg.set_cpt(n, g.cpt(n))
    return tg


def main():
    cells = [("len-ctrl k2 d0", 2, 0), ("len-ctrl k2 d10", 2, 10),
             ("len-ctrl k2 d25", 2, 25), ("len-ctrl k2 d45", 2, 45),
             ("cliff k4 d0", 4, 0), ("cliff k5 d0", 5, 0),
             ("DECISIVE k2 d45 (long,low)", 2, 45), ("DECISIVE k5 d0 (short,high)", 5, 0),
             ("MATCHLEN k2 d120 (>=k5 len, low)", 2, 120)]
    jpath = HERE / "rows.jsonl"
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
            g = k_graph_padded(k, d, 41000 + 1000 * k + 7 * d + i)
            truth = round(InterventionEngine(truth_graph(g, k)).query_intervention(
                "Y", 1, {"X": 1}).value, 6)
            text = R.render(g, k)
            ntok_prompt = len(text.split())          # crude word count as length proxy
            gpt = R.ask_openai(text)
            row = {"label": label, "k": k, "d": d, "i": i, "truth": truth,
                   "prompt_words": ntok_prompt, "gpt51": gpt}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[{label:30} #{i:02d}] words={ntok_prompt:>4} truth={truth:.3f} gpt={gpt.get('pred')}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print("\n=== confound control: accuracy vs (marginalization 2^k) and prompt length ===")
    print(f"{'cell':>30} | {'2^k':>5} | {'prompt words':>12} | {'gpt acc(±.005)':>14}")
    summ = {}
    for label, k, d in cells:
        kr = [r for r in rows if r["label"] == label]
        gp = [r for r in kr if r["gpt51"]["pred"] is not None]
        acc = np.mean([abs(r["gpt51"]["pred"] - r["truth"]) <= TOL for r in gp]) if gp else None
        words = int(np.mean([r["prompt_words"] for r in kr])) if kr else 0
        summ[label] = {"two_k": 2 ** k, "prompt_words": words,
                       "acc": round(acc, 3) if acc is not None else None}
        print(f"{label:>30} | {2**k:>5} | {words:>12} | {str(summ[label]['acc']):>14}")
    (HERE / "results.json").write_text(json.dumps(summ, indent=2))
    print("\nDecisive comparison: if 'k2 d45 (long,low)' >> 'k5 d0 (short,high)' in accuracy")
    print("despite LONGER prompt, the cliff is marginalization load (2^k), NOT token length.")


if __name__ == "__main__":
    main()
