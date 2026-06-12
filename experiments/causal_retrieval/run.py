"""Causal-aware retrieval (frozen prereg 053cbdf). Downstream metric = transfer
bias from applying a retrieved memory's adjustment variable to the query. Wrong
structure -> adjusting a COLLIDER -> bias. Exact via the discrete engine.
Run: python experiments/causal_retrieval/run.py
"""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
D, K, MQ, SEEDS = 32, 60, 40, list(range(20))


def make_case(struct, rng):
    """struct 'a': A confounds, B collider. 'b': B confounds, A collider."""
    conf, coll = ("A", "B") if struct == "a" else ("B", "A")
    g = CausalGraph()
    for n in ("X", "Y", "A", "B"):
        g.add_variable(Variable(n))
    g.add_edge(conf, "X"); g.add_edge(conf, "Y"); g.add_edge("X", "Y")
    g.add_edge("X", coll); g.add_edge("Y", coll)        # collider
    pc = round(float(rng.uniform(.3, .7)), 2)
    g.set_cpt(conf, {(): {1: pc, 0: 1 - pc}})
    # X parents: {conf}
    oX = list(g.parent_order("X"))
    g.set_cpt("X", {c: _b(rng) for c in itertools.product((1, 0), repeat=len(oX))})
    # Y parents: {conf, X}
    oY = list(g.parent_order("Y"))
    g.set_cpt("Y", {c: _b(rng) for c in itertools.product((1, 0), repeat=len(oY))})
    # collider parents: {X, Y}
    oC = list(g.parent_order(coll))
    g.set_cpt(coll, {c: _b(rng) for c in itertools.product((1, 0), repeat=len(oC))})
    return g, conf, coll


def _b(rng):
    p = round(float(rng.uniform(.1, .9)), 3)
    return {1: p, 0: 1 - p}


def true_ate(eng):
    return (eng.query_intervention("Y", 1, {"X": 1}).value
            - eng.query_intervention("Y", 1, {"X": 0}).value)


def adjusted_ate(eng, V):
    """do-ATE estimate adjusting for variable V (backdoor formula with Z={V})."""
    g = eng.g
    out = 0.0
    for v in g.states(V):
        pv = eng.query_observation(V, v, {}).value
        d = (eng.query_observation("Y", 1, {"X": 1, V: v}).value
             - eng.query_observation("Y", 1, {"X": 0, V: v}).value)
        out += pv * d
    return out


def main():
    bias = {"semantic": [], "structural": []}
    collider_misadjust = {"semantic": 0, "structural": 0}
    nq = 0
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # memory bank: structure label + surface embedding (independent)
        mem = []
        for j in range(K):
            st = "a" if j % 2 == 0 else "b"
            emb = rng.standard_normal(D)
            mem.append({"struct": st, "emb": emb / np.linalg.norm(emb)})
        mem_emb = np.stack([m["emb"] for m in mem])
        for _ in range(MQ):
            nq += 1
            qst = "a" if rng.random() < 0.5 else "b"
            qg, qconf, qcoll = make_case(qst, rng)
            qeng = InterventionEngine(qg)
            t = true_ate(qeng)
            qemb = rng.standard_normal(D); qemb /= np.linalg.norm(qemb)
            # semantic: top-1 by cosine (independent of structure)
            sem = mem[int(np.argmax(mem_emb @ qemb))]
            # structural: top-1 sharing structure label
            structn = next(m for m in mem if m["struct"] == qst)
            for name, retr in (("semantic", sem), ("structural", structn)):
                # transfer the retrieved memory's CONFOUNDER LABEL as adjustment var
                adj_var = "A" if retr["struct"] == "a" else "B"
                est = adjusted_ate(qeng, adj_var)
                bias[name].append(abs(est - t))
                if adj_var == qcoll:                  # adjusted a collider
                    collider_misadjust[name] += 1

    res = {"n_queries": nq,
           "bias_median": {k: round(float(np.median(v)), 5) for k, v in bias.items()},
           "bias_mean": {k: round(float(np.mean(v)), 5) for k, v in bias.items()},
           "collider_misadjust_rate": {k: round(v / nq, 3)
                                       for k, v in collider_misadjust.items()},
           "structural_better_count": int(np.sum(
               np.array(bias["structural"]) < np.array(bias["semantic"]))),
           "n_pairs": len(bias["semantic"])}
    res["verdict"] = ("structural_wins"
                      if res["bias_median"]["structural"] < res["bias_median"]["semantic"]
                      and res["structural_better_count"] >= 0.6 * res["n_pairs"]
                      else "no_structural_advantage")
    (HERE / "results.json").write_text(json.dumps(res, indent=2))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()[:16]
    print("=== Causal-aware retrieval (transfer bias, lower=better) ===")
    print(f"queries={nq}")
    print(f"  semantic  : bias median {res['bias_median']['semantic']}  "
          f"collider-misadjust {res['collider_misadjust_rate']['semantic']}")
    print(f"  structural: bias median {res['bias_median']['structural']}  "
          f"collider-misadjust {res['collider_misadjust_rate']['structural']}")
    print(f"structural better in {res['structural_better_count']}/{res['n_pairs']}")
    print(f"VERDICT: {res['verdict']}")
    print(f"results_sha={sha}")


if __name__ == "__main__":
    main()
