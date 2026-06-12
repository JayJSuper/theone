"""Baseline B v1 - graph-scale knob (frozen prereg d5ed61b). PILOT mode default.
Run: source ~/.theone_keys.env && python experiments/baseline_b_v1_scale/run.py [--n 10]
"""
from __future__ import annotations
import argparse
import itertools
import json
import re
import time
import hashlib
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine
from theone.llm import DeepSeekClient

HERE = Path(__file__).parent
BASE_SEED = 20260613
SIZES = {"S": 5, "M": 8, "L": 12}
TOL = 0.005

# ----------------------------------------------------------- DAG generator
def gen_dag(n_nodes: int, rng) -> dict:
    """Random DAG over V0..V{n-1} (topological by index). Forced: X -> Y direct
    edge + one observed confounder U* (parent of both X and Y). Others random
    (<=3 parents). Returns {names, parents{node:[...]}, x, y, cpts}."""
    names = [f"V{i}" for i in range(n_nodes)]
    xi = n_nodes // 3
    yi = n_nodes - 1
    ui = rng.integers(0, xi)                       # confounder before X
    parents = {names[i]: [] for i in range(n_nodes)}
    for i in range(n_nodes):
        if i == 0:
            continue
        cand = [j for j in range(i) if j != yi]
        k = int(rng.integers(0, min(3, len(cand)) + 1))
        ps = sorted(rng.choice(cand, size=k, replace=False).tolist()) if k else []
        parents[names[i]] = [names[j] for j in ps]
    # force U*->X, U*->Y, X->Y
    for (child, par) in ((xi, ui), (yi, ui), (yi, xi)):
        if names[par] not in parents[names[child]]:
            parents[names[child]].append(names[par])
    for k_ in parents:                              # cap at 3 parents (keep forced)
        if len(parents[k_]) > 3:
            forced = {names[ui], names[xi]} if k_ == names[yi] else (
                {names[ui]} if k_ == names[xi] else set())
            keep = [p for p in parents[k_] if p in forced]
            rest = [p for p in parents[k_] if p not in forced]
            parents[k_] = keep + rest[:3 - len(keep)]
    cpts = {}
    for v in names:
        ps = parents[v]
        rows = {}
        for combo in itertools.product((1, 0), repeat=len(ps)):
            rows[combo] = round(float(rng.uniform(0.05, 0.95)), 2)
        cpts[v] = rows
    return {"names": names, "parents": parents, "x": names[xi], "y": names[yi],
            "u": names[ui], "cpts": cpts}


def gen_fixed_skeleton(n_nodes: int, rng) -> dict:
    """Q-C11 (AM-007 sibling, gatekeeper-designed): FIXED single-backdoor skeleton
    + irrelevant distractor nodes. U=V0, X=V1, Y=V2 (adjustment set always {V0});
    V3..V{n-1} are roots connected to nothing. Isolates 'scale / noise tolerance'
    from 'identification difficulty' (which is held constant)."""
    names = [f"V{i}" for i in range(n_nodes)]
    u, x, y = names[0], names[1], names[2]
    parents = {v: [] for v in names}
    parents[x] = [u]
    parents[y] = [u, x]
    # distractors V3.. stay parentless and childless
    cpts = {}
    for v in names:
        ps = parents[v]
        cpts[v] = {combo: round(float(rng.uniform(0.05, 0.95)), 2)
                   for combo in itertools.product((1, 0), repeat=len(ps))}
    return {"names": names, "parents": parents, "x": x, "y": y, "u": u, "cpts": cpts}


def build_graph(d: dict) -> CausalGraph:
    g = CausalGraph()
    for v in d["names"]:
        g.add_variable(Variable(v))
    for v, ps in d["parents"].items():
        for p in ps:
            g.add_edge(p, v)
    for v in d["names"]:
        ps_decl = d["parents"][v]
        order = list(g.parent_order(v))
        rows = {}
        for combo, p1 in d["cpts"][v].items():
            m = dict(zip(ps_decl, combo))
            key = tuple(m[p] for p in order)
            rows[key] = {1: p1, 0: 1 - p1}
        g.set_cpt(v, rows)
    return g


def render_text(d: dict) -> str:
    lines = [f"A system has {len(d['names'])} binary variables: "
             + ", ".join(d["names"]) + "."]
    for v, ps in d["parents"].items():
        if ps:
            lines.append(f"{' and '.join(ps)} directly influence {v}.")
        else:
            lines.append(f"{v} has no parents.")
    lines.append("Measured quantities:")
    for v in d["names"]:
        ps = d["parents"][v]
        for combo, p1 in d["cpts"][v].items():
            cond = ",".join(f"{p}={c}" for p, c in zip(ps, combo))
            lines.append(f"P({v}=1|{cond})={p1}" if ps else f"P({v}=1)={p1}")
    lines.append(f"Question: what is P({d['y']}=1|do({d['x']}=1)), the probability "
                 f"of {d['y']}=1 if {d['x']} were set to 1 by intervention? "
                 f"Give 4 decimals.")
    return "\n".join(lines)


# ----------------------------------------------------------- subjects
PROTO = "\nEnd your reply with exactly one line:\nANSWER: <number with 4 decimals>"
SYS_A = "You answer probability questions about causal systems. Be precise."
SYS_B = ("You are an expert in causal inference. For P(Y|do(X=x)): perform graph "
         "surgery - remove all incoming edges of X, clamp X=x, then compute the "
         "marginal of Y under the truncated factorization "
         "P(v) = prod_i P(v_i | parents(v_i)) with X clamped. Sum over all "
         "configurations of the remaining variables. Work step by step and do "
         "the arithmetic carefully.")
_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)


def ask(client, system, text):
    t0 = time.time()
    try:
        out = client.chat([{"role": "system", "content": system},
                           {"role": "user", "content": text + PROTO}],
                          max_tokens=4096, temperature=0.0)
    except Exception as e:
        return {"pred": None, "latency": time.time() - t0, "tokens": 0,
                "fail": f"transport: {e}"[:120]}
    m = None
    for m in _ANS.finditer(out["content"] or ""):
        pass
    return {"pred": float(m.group(1)) if m else None,
            "latency": round(time.time() - t0, 2),
            "tokens": out["usage"].get("total_tokens", 0),
            "fail": None if m else "no ANSWER line (budget likely exhausted)"}


_P_LINE = re.compile(r"P\((V\d+)=1(?:\|([^)]*))?\)=([0-9]*\.?[0-9]+)")
_INFL = re.compile(r"^(.*) directly influence (V\d+)\.$")


def the_one(text: str) -> dict:
    """C: parse the standardized block -> engine (exact, with receipt)."""
    t0 = time.time()
    parents, cpt_rows = {}, {}
    for line in text.split("\n"):
        mi = _INFL.match(line.strip())
        if mi:
            parents[mi.group(2)] = [p.strip() for p in mi.group(1).split(" and ")]
    qx = re.search(r"do\((V\d+)=1\)", text).group(1)
    qy = re.search(r"what is P\((V\d+)=1\|do", text).group(1)
    names = sorted(set(re.findall(r"V\d+", text)), key=lambda s: int(s[1:]))
    for v in names:
        parents.setdefault(v, [])
        cpt_rows.setdefault(v, {})
    for m in _P_LINE.finditer(text):
        v, cond, p = m.group(1), m.group(2), float(m.group(3))
        if cond:
            combo = tuple(int(c.split("=")[1]) for c in cond.split(","))
        else:
            combo = ()
        cpt_rows[v][combo] = p
    d = {"names": names, "parents": parents, "x": qx, "y": qy,
         "cpts": cpt_rows}
    g = build_graph(d)
    eng = InterventionEngine(g)
    val = eng.query_intervention(qy, 1, {qx: 1}).value
    return {"pred": round(val, 6), "latency": round(time.time() - t0, 4),
            "tokens": 0, "fail": None,
            "receipt": {"graph_hash": g.content_hash(),
                        "method": "graph_surgery_do"}}


# ----------------------------------------------------------- run
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)     # PILOT default
    ap.add_argument("--base-seed", type=int, default=BASE_SEED,
                    help="formal runs use a FRESH seed (burn discipline: pilot "
                         "problems are spent and excluded)")
    ap.add_argument("--skeleton", choices=["random", "fixed"], default="random",
                    help="random=ecological DAG (scale+complexity); "
                         "fixed=Q-C11 single-backdoor + distractors (pure scale)")
    args = ap.parse_args()
    pilot = args.n < 150
    gen = gen_fixed_skeleton if args.skeleton == "fixed" else gen_dag
    suffix = f".{args.skeleton}"
    client = DeepSeekClient(timeout=180)
    jsonl = (HERE / f"rows{suffix}.jsonl").open("a")  # crash-safe incremental log
    rows = []
    for tier, n_nodes in SIZES.items():
        rng = np.random.default_rng(args.base_seed + n_nodes)
        for i in range(args.n):
            d = gen(n_nodes, rng)
            g = build_graph(d)
            truth = InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value
            text = render_text(d)
            a = ask(client, SYS_A, text)
            b = ask(client, SYS_B, text)
            c = the_one(text)
            # C rounds to 6 decimals; allow exactly that rounding, nothing more
            assert abs(c["pred"] - truth) <= 5e-7, "C parser mismatch vs generator"
            row = {"tier": tier, "i": i, "truth": round(truth, 6),
                   "A": a, "B": b, "C": {k: c[k] for k in ("pred", "latency", "fail")}}
            rows.append(row)
            jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
            print(f"[{tier}{i:02d}] truth={truth:.4f}  "
                  f"A={a['pred']} ({a['latency']}s)  B={b['pred']} ({b['latency']}s)  "
                  f"C={c['pred']} ({c['latency']}s)", flush=True)

    summary = {"mode": "PILOT" if pilot else "FORMAL", "n_per_tier": args.n,
               "base_seed": args.base_seed}
    for tier in SIZES:
        tr = [r for r in rows if r["tier"] == tier]
        for s in ("A", "B", "C"):
            preds = [r[s]["pred"] for r in tr]
            errs = [abs(p - r["truth"]) if p is not None else None
                    for p, r in zip(preds, tr)]
            valid = [e for e in errs if e is not None]
            summary[f"{tier}_{s}"] = {
                "acc": round(sum(1 for e in valid if e <= TOL) / len(tr), 3),
                "mae": round(float(np.mean(valid)), 5) if valid else None,
                "protocol_fail": sum(1 for e in errs if e is None),
                "mean_latency_s": round(float(np.mean(
                    [r[s]["latency"] for r in tr])), 2)}
    out = {"summary": summary, "rows": rows}
    (HERE / f"results{suffix}.json").write_text(json.dumps(out, indent=2))
    sha = hashlib.sha256((HERE / f"results{suffix}.json").read_bytes()).hexdigest()[:16]
    print("\n=========== SUMMARY ({} per tier, {}) ===========".format(
        args.n, summary["mode"]))
    print(f"{'tier':>5} {'A acc':>7} {'B acc':>7} {'C acc':>7}   "
          f"{'A mae':>8} {'B mae':>8}   {'A lat':>7} {'B lat':>7} {'C lat':>9}")
    for tier in SIZES:
        s = summary
        print(f"{tier:>5} {s[f'{tier}_A']['acc']:>7} {s[f'{tier}_B']['acc']:>7} "
              f"{s[f'{tier}_C']['acc']:>7}   {s[f'{tier}_A']['mae']:>8} "
              f"{s[f'{tier}_B']['mae']:>8}   {s[f'{tier}_A']['mean_latency_s']:>6}s "
              f"{s[f'{tier}_B']['mean_latency_s']:>6}s "
              f"{s[f'{tier}_C']['mean_latency_s']:>8}s")
    print(f"protocol failures: " + ", ".join(
        f"{t}:{summary[f'{t}_A']['protocol_fail']}A/{summary[f'{t}_B']['protocol_fail']}B"
        for t in SIZES))
    print(f"results_sha={sha}")


if __name__ == "__main__":
    main()
