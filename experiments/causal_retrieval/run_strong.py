"""Causal-aware retrieval, STRONG-COLLIDER regime (WEAK-02 §3 named next step).
WEAK-02 found structural retrieval's value is a safety guarantee (zero collider
mis-adjustment) whose payoff scales with collider-bias severity — small in the
random domain. Here the collider is STRONG (P(collider|X,Y) high-contrast in X,Y),
so mis-adjusting it induces large bias. Prediction (WEAK-02): structural retrieval's
per-query advantage now exceeds the prereg 60% threshold.

Run: python experiments/causal_retrieval/run_strong.py
"""
from __future__ import annotations
import importlib.util, itertools, json
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("cr", HERE / "run.py")
CR = importlib.util.module_from_spec(_s); _s.loader.exec_module(CR)
D, K, MQ, SEEDS = CR.D, CR.K, CR.MQ, CR.SEEDS


def make_case_strong(struct, rng):
    conf, coll = ("A", "B") if struct == "a" else ("B", "A")
    g = CausalGraph()
    for n in ("X", "Y", "A", "B"):
        g.add_variable(Variable(n))
    g.add_edge(conf, "X"); g.add_edge(conf, "Y"); g.add_edge("X", "Y")
    g.add_edge("X", coll); g.add_edge("Y", coll)
    pc = round(float(rng.uniform(.4, .6)), 2)
    g.set_cpt(conf, {(): {1: pc, 0: 1 - pc}})
    # STRONG confounder effect on X and Y (high contrast) so confounding is real
    oX = list(g.parent_order("X"))
    g.set_cpt("X", {c: ({1: 0.85, 0: 0.15} if c[oX.index(conf)] == 1 else {1: 0.15, 0: 0.85})
                    for c in itertools.product((1, 0), repeat=len(oX))})
    oY = list(g.parent_order("Y"))
    def yp(c):
        cf = c[oY.index(conf)]; x = c[oY.index("X")]
        p = 0.10 + 0.40 * cf + 0.40 * x; return {1: round(p, 3), 0: round(1 - p, 3)}
    g.set_cpt("Y", {c: yp(c) for c in itertools.product((1, 0), repeat=len(oY))})
    # STRONG collider: P(coll=1|X,Y) high-contrast in X and Y -> conditioning opens path hard
    oC = list(g.parent_order(coll))
    def cp(c):
        x = c[oC.index("X")]; y = c[oC.index("Y")]
        p = 0.03 + 0.46 * x + 0.46 * y; return {1: round(p, 3), 0: round(1 - p, 3)}
    g.set_cpt(coll, {c: cp(c) for c in itertools.product((1, 0), repeat=len(oC))})
    return g, conf, coll


def main():
    bias = {"semantic": [], "structural": []}; mis = {"semantic": 0, "structural": 0}; nq = 0
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        mem = [{"struct": "a" if j % 2 == 0 else "b",
                "emb": (lambda e: e / np.linalg.norm(e))(rng.standard_normal(D))} for j in range(K)]
        mem_emb = np.stack([m["emb"] for m in mem])
        for _ in range(MQ):
            nq += 1
            qst = "a" if rng.random() < 0.5 else "b"
            qg, qconf, qcoll = make_case_strong(qst, rng)
            qeng = InterventionEngine(qg); t = CR.true_ate(qeng)
            qemb = rng.standard_normal(D); qemb /= np.linalg.norm(qemb)
            sem = mem[int(np.argmax(mem_emb @ qemb))]
            structn = next(m for m in mem if m["struct"] == qst)
            for name, retr in (("semantic", sem), ("structural", structn)):
                adj = "A" if retr["struct"] == "a" else "B"
                bias[name].append(abs(CR.adjusted_ate(qeng, adj) - t))
                if adj == qcoll: mis[name] += 1
    bs, bt = np.array(bias["semantic"]), np.array(bias["structural"])
    better = int(np.sum(bt < bs)); npairs = len(bs)
    res = {"regime": "strong_collider", "n_queries": nq,
           "bias_median": {"semantic": round(float(np.median(bs)), 5), "structural": round(float(np.median(bt)), 5)},
           "bias_mean": {"semantic": round(float(np.mean(bs)), 5), "structural": round(float(np.mean(bt)), 5)},
           "collider_misadjust_rate": {k: round(v / nq, 3) for k, v in mis.items()},
           "structural_better": f"{better}/{npairs}", "structural_better_frac": round(better / npairs, 3),
           "verdict": "structural_wins" if (np.median(bt) < np.median(bs) and better >= 0.6 * npairs) else "no_structural_advantage"}
    (HERE / "results_strong.json").write_text(json.dumps(res, indent=2))
    print("=== Causal-aware retrieval, STRONG-COLLIDER (transfer bias, lower=better) ===")
    print(f"queries={nq}")
    print(f"  semantic  : bias median {res['bias_median']['semantic']} mean {res['bias_mean']['semantic']} collider-misadjust {res['collider_misadjust_rate']['semantic']}")
    print(f"  structural: bias median {res['bias_median']['structural']} mean {res['bias_mean']['structural']} collider-misadjust {res['collider_misadjust_rate']['structural']}")
    print(f"structural better in {res['structural_better']} ({res['structural_better_frac']}) | threshold 0.60")
    print(f"VERDICT: {res['verdict']}")


if __name__ == "__main__":
    main()
