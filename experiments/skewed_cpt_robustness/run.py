"""Q-C25 robustness check (gatekeeper-ruled, paper appendix): does the cliff's
engine advantage depend on the near-0.5 concentration artifact of the original
uniform-CPT generator?

The uniform[.1,.9] symmetric CPT makes true_do -> 0.5 at high k, so a collapsed
LLM guessing ~0.5 gets LUCKY (artificially lowering its high-k error, esp. k=6).
Here Y's CPT is drawn from Beta(5,2) (mean 0.714) -> true_do concentrates at ~0.71,
AWAY from 0.5 -> no lucky shortcut. Prediction (gatekeeper): LLM error at k=6
WORSENS (recovers monotonicity), engine stays 1.000, advantage unchanged or LARGER.

Same SCM structure/seeds as the cliff (only Y-CPT distribution changes). IPRG gate
(AM-016): engine truth independently recomputed by pgmpy.
Run: source ~/.theone_keys.env && python experiments/skewed_cpt_robustness/run.py
"""
from __future__ import annotations
import importlib.util, itertools, json
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)
_c = importlib.util.spec_from_file_location("cfs", HERE.parent / "cpt_finite_sample" / "run.py")
CFS = importlib.util.module_from_spec(_c); _c.loader.exec_module(CFS)

KS = [4, 5, 6]
N = 25
TOL = 0.005


def k_graph_skewed(k, seed, a=5.0, b=2.0):
    rng = np.random.default_rng(seed); g = CausalGraph()
    Us = [f"U{i}" for i in range(k)]
    for n in Us + ["X", "Y", "W"]:
        g.add_variable(Variable(n))
    for u in Us:
        g.add_edge(u, "X"); g.add_edge(u, "Y")
    g.add_edge("X", "Y")
    pw = round(float(rng.uniform(.3, .7)), 2); g.set_cpt("W", {(): {1: pw, 0: round(1 - pw, 2)}})
    for u in Us:
        p = round(float(rng.uniform(.3, .7)), 2); g.set_cpt(u, {(): {1: p, 0: round(1 - p, 2)}})
    oX = list(g.parent_order("X")); rows = {}
    for c in itertools.product((1, 0), repeat=len(oX)):
        p = round(float(rng.uniform(.1, .9)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("X", rows)
    oY = list(g.parent_order("Y")); rows = {}
    for c in itertools.product((1, 0), repeat=len(oY)):
        p = round(float(rng.beta(a, b)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
    g.set_cpt("Y", rows)
    return g


def main():
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["k"], r["i"]))
    jf = jpath.open("a"); iprg_max = 0.0
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            g = k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
            pg = CFS.pgmpy_do(g, k); iprg_max = max(iprg_max, abs(pg - truth))
            text = R.render(g, k)
            gpt = R.ask_openai(text); ds = R.ask_deepseek(text)
            row = {"k": k, "i": i, "combos": 2 ** k, "truth": truth,
                   "guess05_err": round(abs(0.5 - truth), 6),
                   "gpt51": gpt, "deepseek": ds, "iprg": round(abs(pg - truth), 10)}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.3f} gpt={gpt.get('pred')} ds={ds.get('pred')} iprg={abs(pg-truth):.1e}", flush=True)
    # summary vs uniform cliff
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print(f"\nIPRG max|pgmpy-engine|={iprg_max:.2e} -> {'PASS' if iprg_max<1e-6 else 'FAIL'}")
    print(f"{'k':>3} {'truth~':>7} {'gpt_acc':>8} {'gpt_err':>8} {'ds_acc':>7} {'ds_fail':>8}  (engine acc=1.000 by construction)")
    summ = {}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]
        gp = [r for r in kr if r["gpt51"]["pred"] is not None]
        gacc = np.mean([1 if abs(r["gpt51"]["pred"] - r["truth"]) <= TOL else 0 for r in gp]) if gp else None
        gerr = np.mean([abs(r["gpt51"]["pred"] - r["truth"]) for r in gp]) if gp else None
        dp = [r for r in kr if r["deepseek"]["pred"] is not None]
        dacc = np.mean([1 if abs(r["deepseek"]["pred"] - r["truth"]) <= TOL else 0 for r in dp]) if dp else None
        dfail = sum(1 for r in kr if r["deepseek"]["pred"] is None)
        summ[k] = {"truth_mean": round(float(np.mean([r["truth"] for r in kr])), 3),
                   "gpt_acc": round(float(gacc), 3) if gacc is not None else None,
                   "gpt_mae": round(float(gerr), 4) if gerr is not None else None,
                   "ds_acc": round(float(dacc), 3) if dacc is not None else None,
                   "ds_protocol_fail": f"{dfail}/{len(kr)}"}
        print(f"{k:>3} {summ[k]['truth_mean']:>7} {str(summ[k]['gpt_acc']):>8} {str(summ[k]['gpt_mae']):>8} {str(summ[k]['ds_acc']):>7} {summ[k]['ds_protocol_fail']:>8}")
    (HERE / "results.json").write_text(json.dumps({"summary": summ, "iprg_max": iprg_max, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
