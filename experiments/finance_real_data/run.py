"""finance_real_data — auditable causal attribution + credential on REAL public data.

A first step from synthetic CPTs to real data: the German Credit dataset
(sklearn `credit-g`, 1000 rows). We binarize into a small discrete SCM whose
STRUCTURE is a DOMAIN ASSUMPTION (not discovered from data), estimate the CPTs
from the real frequencies (Laplace-smoothed), and have the engine compute
P(Y=1|do(X=x)) with an auditable CREDENTIAL (adjustment set, backdoor paths,
independent pgmpy recompute, regime statement). We then feed the SAME
real-data-derived CPTs to gpt-5.1 in words and check whether it reproduces the
engine's exact value.

HONEST FRAMING: this validates "the engine certifies a CALCULATION on a
real-data SCM", NOT "we discovered the causal structure from data". The graph is
an assumption; the sample is real but small (n=1000).

Run: source ~/.theone_keys.env && .venv/bin/python experiments/finance_real_data/run.py
Offline (no LLM): .venv/bin/python experiments/finance_real_data/run.py --offline
"""
from __future__ import annotations
import importlib.util, itertools, json, os, re, sys, time, urllib.request
from pathlib import Path
import numpy as np

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
# reuse the frozen, independently-validated pgmpy translator/surgery
_o = importlib.util.spec_from_file_location("scaleoracle", ROOT / "experiments" / "oracle_crosscheck" / "scale_oracle.py")
O = importlib.util.module_from_spec(_o); _o.loader.exec_module(O)

LAPLACE = 1.0           # additive smoothing on every CPT cell
# treatment X, outcome Y, and the 3 assumed binary confounders
CONF = ["Semp", "Shou", "Ssav"]
CONF_DESC = {
    "Semp": "unstable employment (employed <1yr or unemployed)",
    "Shou": "non-owned housing (rent / for free)",
    "Ssav": "low savings (checking-account savings <100 DM)",
}
PROTO = "\nEnd your reply with exactly one line:\nANSWER: <number with 4 decimals>"
SYS_FIN = ("You are a quantitative model-risk expert. For an interventional query "
           "P(Y=1|do(X=x)), use back-door adjustment over ALL listed confounders: "
           "P(Y=1|do(X=x)) = sum over every confounder configuration u of "
           "P(Y=1|X=x,u) * P(u), where P(u) is the PRODUCT of the confounders' "
           "marginal base rates (they are mutually independent roots here). "
           "Work carefully and arithmetically.")
_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)


def _last(content):
    m = None
    for m in _ANS.finditer(content or ""):
        pass
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 1. DATA + binarization
# ---------------------------------------------------------------------------
def load_binarized():
    """Returns (DataFrame of binary columns, median credit_amount) or raises."""
    from sklearn.datasets import fetch_openml
    import pandas as pd
    df = fetch_openml("credit-g", version=1, as_frame=True).frame
    med = float(df["credit_amount"].median())
    B = pd.DataFrame({
        "Y": (df["class"] == "bad").astype(int),                       # default
        "X": (df["credit_amount"] > med).astype(int),                  # high loan
        "Semp": df["employment"].isin(["<1", "unemployed"]).astype(int),
        "Shou": (df["housing"] != "own").astype(int),
        "Ssav": (df["savings_status"] == "<100").astype(int),
    })
    return B, med, len(df)


# ---------------------------------------------------------------------------
# 2+3. ASSUMED structure + CPTs estimated from real frequencies
# ---------------------------------------------------------------------------
def build_scm(B):
    """Structure is a DOMAIN ASSUMPTION: each confounder S is a common cause of
    X and Y; X->Y. CPTs are Laplace-smoothed frequency estimates from real data."""
    g = CausalGraph()
    for n in ["X", "Y"] + CONF:
        g.add_variable(Variable(n))
    for s in CONF:
        g.add_edge(s, "X"); g.add_edge(s, "Y")
    g.add_edge("X", "Y")

    def est(col, parents):
        order = sorted(parents); rows = {}
        if not order:
            n1 = int(B[col].sum()); n = len(B)
            p1 = (n1 + LAPLACE) / (n + 2 * LAPLACE)
            return {(): {1: p1, 0: 1 - p1}}, {(): {"n": n, "n1": n1}}
        counts = {}
        for combo in itertools.product((1, 0), repeat=len(order)):
            mask = np.ones(len(B), bool)
            for p, c in zip(order, combo):
                mask &= (B[p] == c).values
            sub = B[col][mask]; n1 = int(sub.sum()); n = int(len(sub))
            p1 = (n1 + LAPLACE) / (n + 2 * LAPLACE)
            rows[combo] = {1: p1, 0: 1 - p1}; counts[combo] = {"n": n, "n1": n1}
        return rows, counts

    support = {}
    for s in CONF:
        rows, c = est(s, []); g.set_cpt(s, rows); support[s] = c
    rows, c = est("X", CONF); g.set_cpt("X", rows); support["X"] = c
    rows, c = est("Y", CONF + ["X"]); g.set_cpt("Y", rows); support["Y"] = c
    return g, support


# ---------------------------------------------------------------------------
# 4. IPRG: independent pgmpy recompute of do(X=v)
# ---------------------------------------------------------------------------
def pgmpy_do(g, x, y, value):
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    m = O.to_pgmpy(g)
    for p in list(g.parent_order(x)):
        m.remove_edge(p, x)
    m.remove_cpds(m.get_cpds(x))
    col = [[0.0], [1.0]] if value == 1 else [[1.0], [0.0]]
    m.add_cpds(TabularCPD(x, 2, col, state_names={x: [0, 1]}))
    assert m.check_model()
    return float(VariableElimination(m).query([y], show_progress=False).values[1])


# ---------------------------------------------------------------------------
# 3b. CREDENTIAL
# ---------------------------------------------------------------------------
def build_credential(g, do_value, engine_val, pgmpy_val, n_rows):
    return {
        "query": f"P(Y=1 | do(X={do_value}))",
        "estimand": "back-door adjustment",
        "adjustment_set": list(CONF),
        "backdoor_paths_blocked": [f"X <- {s} -> Y" for s in CONF],
        "engine_value": round(engine_val, 6),
        "independent_recompute": {"oracle": f"pgmpy {__import__('pgmpy').__version__}",
                                  "value": round(pgmpy_val, 6),
                                  "abs_diff": round(abs(engine_val - pgmpy_val), 12),
                                  "agree_1e-6": abs(engine_val - pgmpy_val) < 1e-6},
        "regime_statement": ("STRUCTURE IS A DOMAIN ASSUMPTION (not discovered); "
                             f"CPTs are Laplace({LAPLACE}) frequency estimates from "
                             f"real German-Credit data, n={n_rows}. The credential "
                             "certifies the CALCULATION, not the causal structure."),
    }


# ---------------------------------------------------------------------------
# 5. LLM rendering + call
# ---------------------------------------------------------------------------
def render_for_llm(g, do_value):
    L = [("A credit-default model (from real German-Credit data, binarized) has "
          "3 binary confounders, each a COMMON CAUSE of both the treatment and the "
          "outcome:")]
    for s in CONF:
        L.append(f"  {s} = {CONF_DESC[s]}")
    L.append("Treatment X = high loan amount (credit_amount above the median).")
    L.append("Outcome   Y = loan default (class = bad).")
    L.append("Causal structure (assumed): each S_i -> X and S_i -> Y; and X -> Y. "
             "The S_i are mutually independent root causes.")
    L.append("\nEstimated probabilities (from the real data):")
    for v in ["Semp", "Shou", "Ssav", "X", "Y"]:
        ps = list(g.parent_order(v))
        for combo, d in g.cpt(v).items():
            if ps:
                cond = ",".join(f"{p}={c}" for p, c in zip(ps, combo))
                L.append(f"  P({v}=1|{cond})={d[1]:.4f}")
            else:
                L.append(f"  P({v}=1)={d[1]:.4f}")
    L.append(f"\nQuestion: in a stress test we SET the loan to high, do(X={do_value}). "
             f"What is P(Y=1 | do(X={do_value})) — the causal effect of a high loan "
             "on default, correctly adjusting for the confounders? Give 4 decimals.")
    return "\n".join(L)


def ask_gpt(text, maxtok=8192):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": SYS_FIN},
                {"role": "user", "content": text + PROTO}],
                "max_completion_tokens": maxtok}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]
        tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:120]}
    return {"pred": _last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if _last(c) is not None else "no ANSWER"}


# ---------------------------------------------------------------------------
def main():
    offline = "--offline" in sys.argv
    n_llm = 6  # repeated gpt trials (n<=10) for stability, per do-value

    # 1. data
    try:
        B, med, n_rows = load_binarized()
    except Exception as e:
        print(f"DATA LOAD FAILED: {e}\nStopping honestly — no real data, no run.")
        (HERE / "results.json").write_text(json.dumps({"error": f"data load failed: {e}"}, indent=2))
        return
    print(f"[data] credit-g loaded: n={n_rows}, median credit_amount={med:.0f}")

    # 2+3. SCM
    g, support = build_scm(B)

    # 3+4. engine do() + IPRG, both do-values
    iprg_max = 0.0; do_results = {}
    eng = InterventionEngine(g)
    for dv in (0, 1):
        ev = eng.query_intervention("Y", 1, {"X": dv}).value
        pv = pgmpy_do(g, "X", "Y", dv)
        iprg_max = max(iprg_max, abs(ev - pv))
        cred = build_credential(g, dv, ev, pv, n_rows)
        do_results[dv] = {"engine": round(ev, 6), "pgmpy": round(pv, 6),
                          "iprg_diff": round(abs(ev - pv), 12), "credential": cred}
        print(f"[engine] do(X={dv}): P(Y=1)={ev:.6f}  pgmpy={pv:.6f}  diff={abs(ev-pv):.2e}")
    ate = do_results[1]["engine"] - do_results[0]["engine"]
    iprg_pass = iprg_max < 1e-6
    print(f"[IPRG] max|engine-pgmpy| = {iprg_max:.2e} -> {'PASS' if iprg_pass else 'FAIL'}")
    print(f"[effect] interventional ATE = {ate:+.4f}")

    if not iprg_pass:
        print("IPRG FAILED — refusing to proceed to LLM. Reporting honestly.")
        (HERE / "results.json").write_text(json.dumps(
            {"iprg_max": iprg_max, "iprg_pass": False, "do_results": do_results}, indent=2))
        return

    # 5. LLM on real-data CPT (checkpoint-resumable)
    rowpath = HERE / "rows.jsonl"
    done = set()
    if rowpath.exists():
        for l in rowpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["do"], r["trial"]))
    gpt_rows = [json.loads(l) for l in rowpath.read_text().splitlines() if l.strip()] if rowpath.exists() else []

    if not offline and "OPENAI_API_KEY" in os.environ:
        jf = rowpath.open("a")
        for dv in (0, 1):
            truth = do_results[dv]["engine"]
            for t in range(n_llm):
                if (dv, t) in done:
                    continue
                r = ask_gpt(render_for_llm(g, dv))
                row = {"do": dv, "trial": t, "truth": round(truth, 6), "gpt51": r}
                jf.write(json.dumps(row) + "\n"); jf.flush(); gpt_rows.append(row)
                print(f"[gpt do={dv} t{t}] pred={r.get('pred')} truth={truth:.4f} "
                      f"fail={r.get('fail')}", flush=True)
        jf.close()
    else:
        print("[gpt] skipped (offline or no OPENAI_API_KEY).")

    # summary of gpt vs engine
    TOL = 0.005
    gpt_summary = {}
    for dv in (0, 1):
        rs = [r for r in gpt_rows if r["do"] == dv and r["gpt51"]["pred"] is not None]
        truth = do_results[dv]["engine"]
        if rs:
            errs = [abs(r["gpt51"]["pred"] - truth) for r in rs]
            gpt_summary[dv] = {"n": len(rs), "truth": round(truth, 4),
                               "acc": round(float(np.mean([e <= TOL for e in errs])), 3),
                               "mae": round(float(np.mean(errs)), 4),
                               "preds": [r["gpt51"]["pred"] for r in rs]}
        else:
            gpt_summary[dv] = {"n": 0, "truth": round(truth, 4), "acc": None,
                               "mae": None, "preds": []}
        print(f"[gpt vs engine] do={dv}: truth={truth:.4f} "
              f"acc={gpt_summary[dv]['acc']} mae={gpt_summary[dv]['mae']} n={gpt_summary[dv]['n']}")

    results = {
        "data": {"source": "sklearn fetch_openml('credit-g', version=1)",
                 "n_rows": n_rows, "median_credit_amount": med,
                 "binarization": {
                     "Y": "default = (class == 'bad')",
                     "X": "high loan = (credit_amount > median)",
                     "Semp": CONF_DESC["Semp"], "Shou": CONF_DESC["Shou"],
                     "Ssav": CONF_DESC["Ssav"]}},
        "assumed_structure": {"edges": [f"{s}->X" for s in CONF] + [f"{s}->Y" for s in CONF] + ["X->Y"],
                              "adjustment_set": CONF,
                              "disclaimer": "structure is a DOMAIN ASSUMPTION, not discovered from data"},
        "cpt_support_counts": {k: {str(kk): vv for kk, vv in v.items()} for k, v in support.items()},
        "estimated_cpt": {v: {str(c): round(d[1], 6) for c, d in g.cpt(v).items()}
                          for v in ["Semp", "Shou", "Ssav", "X", "Y"]},
        "do_results": do_results, "interventional_ate": round(ate, 6),
        "iprg_max": iprg_max, "iprg_pass": iprg_pass,
        "gpt_vs_engine": gpt_summary, "tol": TOL,
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\n[done] results.json written. IPRG {'PASS' if iprg_pass else 'FAIL'} ({iprg_max:.2e}).")


if __name__ == "__main__":
    main()
