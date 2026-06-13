"""Bet ① ECOLOGICAL VALIDITY: does the engine's high-k advantage survive when the
CPT is ESTIMATED from finite data instead of given exactly? (Jack ruling NOTE-003.)

Design (frozen NOTE-003 + fairness gate 1.1): reuse the frozen complexity_axis
SCMs. For each, ancestral-sample n∈{50,200,1000} rows, MLE-estimate every CPT
(Laplace add-α), build the ESTIMATED graph. BOTH engine and LLM receive the SAME
estimated CPT table (never raw data — that would be causal discovery, not
inference). Score vs the TRUE do(X=1).

  engine_err  = |engine.do(EST) - true_do|        # irreducible estimation floor
  llm_err     = |llm.answer(EST) - true_do|        # floor + reasoning error
  reasoning   = |llm.answer(EST) - engine.do(EST)| # reasoning error, isolated

Hypothesis: at high k the LLM's reasoning collapse (cliff) persists ON TOP of the
estimation floor -> engine advantage survives CPT noise. Honest counter-risk: at
high k + low n, 2^k parent configs are data-starved -> the estimation floor itself
explodes (curse of dimensionality), a limit no inference method escapes.

IPRG gate (AM-016): engine.do(EST) is independently recomputed by pgmpy.
Run: source ~/.theone_keys.env && python experiments/cpt_finite_sample/run.py
"""
from __future__ import annotations
import importlib.util, json, itertools
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_s); _s.loader.exec_module(R)

KS = [1, 2, 3, 4, 5, 6]
NS = [50, 200, 1000]
N_ENGINE = 120      # instances per (k,n) for the free engine-arm statistics
N_LLM = 20          # instances per (k,n) that also get LLM calls
ALPHA = 1.0         # Laplace smoothing
TOL = 0.005


def topo(g):
    order, done = [], set()
    names = list(g.variables)
    while len(order) < len(names):
        for v in names:
            if v in done:
                continue
            if all(p in done for p in g.parent_order(v)):
                order.append(v); done.add(v)
    return order


def sample_data(g, n, seed):
    rng = np.random.default_rng(seed)
    cols = {v: np.zeros(n, int) for v in g.variables}
    order = topo(g)
    for t in range(n):
        vals = {}
        for v in order:
            key = tuple(vals[p] for p in g.parent_order(v))
            p1 = g.cpt(v)[key][1]
            vals[v] = 1 if rng.random() < p1 else 0
        for v in order:
            cols[v][t] = vals[v]
    return cols


def estimate_graph(g_true, data, alpha=ALPHA):
    g = CausalGraph()
    for v in g_true.variables:
        g.add_variable(Variable(v))
    for v in g_true.variables:
        for p in g_true.parent_order(v):
            g.add_edge(p, v)
    for v in g_true.variables:
        order = g.parent_order(v)
        cpt = {}
        for key in g_true.cpt(v).keys():
            if order:
                mask = np.ones(len(data[v]), bool)
                for i, p in enumerate(order):
                    mask &= (data[p] == key[i])
            else:
                mask = np.ones(len(data[v]), bool)
            tot = int(mask.sum()); ones = int((data[v][mask] == 1).sum())
            p1 = (ones + alpha) / (tot + 2 * alpha)
            cpt[key] = {1: round(p1, 9), 0: round(1 - p1, 9)}
        g.set_cpt(v, cpt)
    return g


# ---- pgmpy independent recompute (IPRG / AM-016) ----
def pgmpy_do(g_est, k):
    from pgmpy.models import DiscreteBayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import CausalInference
    Us = [f"U{i}" for i in range(k)]
    edges = [(u, "X") for u in Us] + [(u, "Y") for u in Us] + [("X", "Y")]
    m = DiscreteBayesianNetwork(edges); m.add_node("W")
    cpds = []
    for v in g_est.variables:
        order = list(g_est.parent_order(v)); sn = {v: [0, 1]}
        for p in order:
            sn[p] = [0, 1]
        if not order:
            p1 = g_est.cpt(v)[()][1]
            cpds.append(TabularCPD(v, 2, [[1 - p1], [p1]], state_names={v: [0, 1]}))
        else:
            combos = list(itertools.product([0, 1], repeat=len(order)))
            r1 = [g_est.cpt(v)[c][1] for c in combos]; r0 = [g_est.cpt(v)[c][0] for c in combos]
            cpds.append(TabularCPD(v, 2, [r0, r1], evidence=order,
                                   evidence_card=[2]*len(order), state_names=sn))
    m.add_cpds(*cpds); assert m.check_model()
    return float(CausalInference(m).query(["Y"], do={"X": 1}, show_progress=False).values[1])


def main():
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["k"], r["n"], r["i"]))
    jf = jpath.open("a")
    iprg_max = 0.0
    for k in KS:
        for n in NS:
            for i in range(N_ENGINE):
                if (k, n, i) in done:
                    continue
                seed = R.BASE_SEED + 1000 * k + i      # SAME true SCM as cliff
                g_true = R.k_graph(k, seed)
                true_do = round(InterventionEngine(g_true).query_intervention("Y", 1, {"X": 1}).value, 6)
                data = sample_data(g_true, n, seed * 100 + n)
                g_est = estimate_graph(g_true, data)
                eng_est = round(InterventionEngine(g_est).query_intervention("Y", 1, {"X": 1}).value, 6)
                row = {"k": k, "n": n, "i": i, "true_do": true_do, "engine_est": eng_est,
                       "engine_err": round(abs(eng_est - true_do), 6)}
                if i < N_LLM:
                    # IPRG gate (AM-016): pgmpy independently recomputes engine-on-estimated
                    pg = pgmpy_do(g_est, k); iprg_max = max(iprg_max, abs(pg - eng_est))
                    row["iprg_pgmpy_diff"] = round(abs(pg - eng_est), 10)
                    text = R.render(g_est, k)
                    gpt = R.ask_openai(text); ds = R.ask_deepseek(text)
                    row["gpt51"] = gpt; row["deepseek"] = ds
                jf.write(json.dumps(row) + "\n"); jf.flush()
                if i < N_LLM:
                    print(f"[k{k} n{n} #{i:02d}] true={true_do:.3f} eng_est={eng_est:.3f} "
                          f"eng_err={abs(eng_est-true_do):.3f} gpt={gpt.get('pred')} ds={ds.get('pred')} "
                          f"iprg={abs(pg-eng_est):.1e}", flush=True)
            print(f"== k={k} n={n} done (engine arm {N_ENGINE}, IPRG max so far {iprg_max:.1e}) ==", flush=True)
    print(f"\nIPRG GATE (AM-016): max|pgmpy-engine on estimated CPT| = {iprg_max:.2e} "
          f"-> {'PASS' if iprg_max < 1e-6 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
