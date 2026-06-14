"""Financial beachhead (Jay's choice: financial risk). Causal risk attribution in
the genuinely-DISCRETE/COMBINATORIAL subdomain where the engine's advantage is real:
credit default under systemic stress factors. (Linear-Gaussian factor returns have
closed-form do-effects — no cliff; default/distress are binary and many systemic
factors → real 2^k marginalization. See Q-C29.)

Structure: k binary systemic stress factors S_i (rate shock, liquidity freeze, …)
each a common cause of firm distress D and default Y; D→Y. Query: P(default=1 |
do(distress=1)) — the causal effect of distress on default under a stress
intervention, adjusting for systemic confounders. Bonus realism: tail events are
RARE (P(S_i=1)∈[0.03,0.20]) → true_do is naturally de-anchored (no anchoring
artifact, for free).

Subjects: gpt-5.1 (financial framing) vs engine C. IPRG gate (pgmpy). AM-007 scoring.
Run: source ~/.theone_keys.env && python experiments/finance_credit_risk/run.py
"""
from __future__ import annotations
import importlib.util, itertools, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_o = importlib.util.spec_from_file_location("scaleoracle", HERE.parent / "oracle_crosscheck" / "scale_oracle.py")
O = importlib.util.module_from_spec(_o); _o.loader.exec_module(O)  # general pgmpy translator + surgery

KS = [4, 5, 6]
N = 25
TOL = 0.005
FACTORS = ["a rate shock", "a liquidity freeze", "credit-spread widening",
           "an equity-market crash", "an FX shock", "sovereign stress"]


def credit_scm(k, seed):
    rng = np.random.default_rng(seed); g = CausalGraph()
    Ss = [f"S{i}" for i in range(k)]
    for n in Ss + ["D", "Y", "Z"]:
        g.add_variable(Variable(n))
    for s in Ss:
        g.add_edge(s, "D"); g.add_edge(s, "Y")
    g.add_edge("D", "Y")
    pz = round(float(rng.uniform(.3, .7)), 2); g.set_cpt("Z", {(): {1: pz, 0: round(1 - pz, 2)}})
    for s in Ss:
        p = round(float(rng.uniform(.03, .20)), 3); g.set_cpt(s, {(): {1: p, 0: round(1 - p, 3)}})
    for v in ("D", "Y"):
        order = list(g.parent_order(v)); rows = {}
        for c in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(.1, .9)), 3); rows[c] = {1: p, 0: round(1 - p, 3)}
        g.set_cpt(v, rows)
    return g


def render_credit(g, k):
    Ss = [f"S{i}" for i in range(k)]
    names = {f"S{i}": FACTORS[i] for i in range(k)}
    L = [f"A credit-risk model has {k} binary systemic stress factors "
         f"({', '.join(Ss)}), namely: " + "; ".join(f"{s}={names[s]}" for s in Ss) + "."]
    L.append("Each systemic factor, when it occurs, raises BOTH a firm's financial "
             "distress (D) and its default (Y): they are common causes (confounders) "
             "of the distress→default relationship. Firm distress D also directly "
             "affects default Y. Z is an unrelated indicator.")
    L.append("Base rates (rare tail events):")
    for v in g.variables:
        ps = list(g.parent_order(v))
        for combo, d in g.cpt(v).items():
            cond = ",".join(f"{p}={c}" for p, c in zip(ps, combo))
            L.append(f"P({v}=1|{cond})={d[1]:.3f}" if ps else f"P({v}=1)={d[1]:.3f}")
    L.append("Question: in a stress test we SET firm distress to present, do(D=1). "
             "What is P(Y=1 | do(D=1)) — the causal effect of distress on default "
             "under this intervention, correctly adjusting for the systemic "
             "confounders? Give 4 decimals.")
    return "\n".join(L)


SYS_FIN = ("You are a quantitative model-risk expert. For an interventional query "
           "P(Y|do(D=d)), use the back-door adjustment over ALL systemic confounders: "
           "P(Y=1|do(D=1)) = sum over all factor configurations s of P(Y=1|D=1,s) * "
           "P(s). Work carefully.")


def ask_gpt(text, maxtok=4096):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": SYS_FIN},
                {"role": "user", "content": text + R.PROTO}],
                "max_completion_tokens": maxtok}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]; tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:80]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                done.add((json.loads(l)["k"], json.loads(l)["i"]))
    jf = jpath.open("a"); iprg_max = 0.0
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            g = credit_scm(k, 7000 + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"D": 1}).value, 6)
            pg = O.pgmpy_do1(g, "D", "Y"); iprg_max = max(iprg_max, abs(pg - truth))
            gpt = ask_gpt(render_credit(g, k))
            row = {"k": k, "i": i, "truth": truth, "gpt51": gpt, "iprg": round(abs(pg - truth), 10)}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.3f} gpt={gpt.get('pred')} iprg={abs(pg-truth):.1e}", flush=True)
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print(f"\nIPRG max|pgmpy-engine|={iprg_max:.2e} -> {'PASS' if iprg_max<1e-6 else 'FAIL'}")
    print(f"=== credit-risk causal attribution: gpt-5.1 vs engine (engine=1.000 by construction) ===")
    summ = {}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]; gp = [r for r in kr if r["gpt51"]["pred"] is not None]
        acc = np.mean([abs(r["gpt51"]["pred"] - r["truth"]) <= TOL for r in gp]) if gp else None
        mae = np.mean([abs(r["gpt51"]["pred"] - r["truth"]) for r in gp]) if gp else None
        fail = sum(1 for r in kr if r["gpt51"]["pred"] is None)
        summ[k] = {"gpt_acc": round(acc, 3) if acc is not None else None,
                   "gpt_mae": round(mae, 4) if mae is not None else None, "fail": f"{fail}/{len(kr)}"}
        print(f"k={k} (2^{k}={2**k}): gpt acc={summ[k]['gpt_acc']} mae={summ[k]['gpt_mae']} fail={summ[k]['fail']} | engine 1.000")
    (HERE / "results.json").write_text(json.dumps({"summary": summ, "iprg_max": iprg_max, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
