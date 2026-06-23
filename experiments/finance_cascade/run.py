"""Multi-debtor DEFAULT CASCADE (stronger finance case study; companion to
finance_credit_risk). A genuinely-discrete, combinatorial SCM where the engine's
2^k marginalization advantage is real and maps directly onto 2008-style CORRELATED
DEFAULT / contagion.

Structure (all binary; total nodes = 2 macro + 2 industry + 1 lead + N debtors):
  - F0,F1   : M=2 common MACRO factors (rare tail events, P in [0.03,0.20]) ->
              systemic confounders. Naturally de-anchored (rare -> joint mass
              concentrates -> true_do spreads, no anchoring artifact, for free).
  - G0,G1   : 2 INDUSTRY factors, each a CHILD of BOTH macro factors
              (F -> G): industry stress is driven by systemic stress.
  - D       : LEAD debtor's distress = the CONTAGION SOURCE. Parents = F0,F1
              (macro confounds the lead's distress).
  - Y0..Y3  : N=4 downstream debtors' binary DEFAULT, 2 per industry.
              Parents(Y_j) = { D, G_{industry(j)} }  -> contagion edge D->Y_j
              PLUS the industry factor. The systemic confounding of D and each
              Y_j is the back-door path  D <- F -> G -> Y_j .

Query: P(Y_j=1 | do(D=1)) for a downstream debtor, AND the combined/portfolio
default rate, with the lead's distress SET (stress test). This is the CAUSAL
effect of default contagion, adjusting away the systemic (macro+industry)
confounders. Exactly the 2008 question: when the bellwether goes, how much of the
co-movement is *causal contagion* vs *shared systemic exposure*?

Node budget: 2 + 2 + 1 + 4 = 9 binary nodes -> 2^9 = 512 joint configs (<2s).
The engine enumerates the FULL joint (O(2^nodes)); 9 nodes is safely inside budget,
so no ancestor-subgraph pruning is needed here (kept under the <=12 hard cap).

Discipline:
  - IPRG: every engine truth independently recomputed by pgmpy; require <1e-6.
  - File-backed, flushed jsonl + resume.
  - LLM (gpt-5.1) contrast: does it crash on the contagion cascade as the
    confounder count (2^k) grows? We sweep difficulty by # systemic confounders
    on the back-door path (macro-only -> macro+industry -> macro+industry+more
    debtors marginalized for the portfolio rate).

Synthetic prototype, NOT real portfolio data. Run:
  source ~/.theone_keys.env && python experiments/finance_cascade/run.py
"""
from __future__ import annotations
import importlib.util, itertools, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent

# pgmpy translator reuse (general binary-DAG -> pgmpy) for the IPRG cross-check
_o = importlib.util.spec_from_file_location(
    "scaleoracle", HERE.parent / "oracle_crosscheck" / "scale_oracle.py")
O = importlib.util.module_from_spec(_o); _o.loader.exec_module(O)

from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

N_PER_CELL = 10          # SCMs per difficulty cell (small per task brief)
TOL = 0.005
MAXTOK = 6144
BASE_SEED = 53000

MACRO = ["a global liquidity freeze", "a credit-spread blowout"]
INDUSTRY = ["the financials sector", "the housing/construction sector"]


# ---------------------------------------------------------------------------
# SCM construction
# ---------------------------------------------------------------------------
def cascade_scm(n_debtors, seed, n_macro=2):
    """Multi-debtor cascade. n_debtors downstream Y_j (split across 2 industries).
    Returns (graph, meta). Nodes = 2 macro + 2 industry + 1 lead D + n_debtors."""
    rng = np.random.default_rng(seed)
    g = CausalGraph()
    Fs = [f"F{i}" for i in range(n_macro)]
    Gs = ["G0", "G1"]
    Ys = [f"Y{j}" for j in range(n_debtors)]
    industry_of = {f"Y{j}": j % 2 for j in range(n_debtors)}  # alternate sectors
    for n in Fs + Gs + ["D"] + Ys:
        g.add_variable(Variable(n))
    # macro -> industry
    for gf in Gs:
        for f in Fs:
            g.add_edge(f, gf)
    # macro -> lead distress (confounds D with each Y via F->G->Y)
    for f in Fs:
        g.add_edge(f, "D")
    # contagion + industry exposure into each debtor
    for j in range(n_debtors):
        y = f"Y{j}"
        g.add_edge("D", y)               # contagion edge (the causal effect we ask)
        g.add_edge(f"G{industry_of[y]}", y)

    # CPTs -------------------------------------------------------------------
    # macro: rare tail events
    for f in Fs:
        p = round(float(rng.uniform(.03, .20)), 3)
        g.set_cpt(f, {(): {1: p, 0: round(1 - p, 3)}})
    # industry, lead, debtors: random conditional tables over parent configs
    for v in Gs + ["D"] + Ys:
        order = list(g.parent_order(v))
        rows = {}
        for c in itertools.product((1, 0), repeat=len(order)):
            p = round(float(rng.uniform(.05, .9)), 3)
            rows[c] = {1: p, 0: round(1 - p, 3)}
        g.set_cpt(v, rows)
    meta = {"Fs": Fs, "Gs": Gs, "Ys": Ys, "industry_of": industry_of,
            "n_nodes": len(Fs) + len(Gs) + 1 + n_debtors}
    return g, meta


# ---------------------------------------------------------------------------
# IPRG: independent pgmpy recomputation of do(D=1) for a target (single Y or
# portfolio default rate = mean_j P(Y_j=1|do(D=1)), which by linearity equals
# the engine's per-debtor average). We cross-check each per-debtor do-effect.
# ---------------------------------------------------------------------------
def pgmpy_do1_target(g, target):
    """P(target=1 | do(D=1)) via independent pgmpy surgery: sever D's incoming
    edges, pin D=1, query target. Mirrors O.pgmpy_do1 but reused for any target."""
    return O.pgmpy_do1(g, "D", target)


# ---------------------------------------------------------------------------
# Rendering for the LLM
# ---------------------------------------------------------------------------
def render(g, meta, query_target, portfolio):
    Fs, Gs, Ys = meta["Fs"], meta["Gs"], meta["Ys"]
    io = meta["industry_of"]
    L = [f"A bank models default CONTAGION across {len(Ys)} borrowers using a "
         "discrete causal network. All variables are binary (1=event occurs)."]
    L.append("Systemic MACRO factors (rare tail events), each a common cause: "
             + ", ".join((f"{f}={MACRO[i]}" if i < len(MACRO) else f"{f}=systemic stress factor {i}")
                         for i, f in enumerate(Fs)) + ".")
    L.append(f"INDUSTRY-stress factors, each driven by BOTH macro factors: "
             f"{Gs[0]}=stress in {INDUSTRY[0]}, {Gs[1]}=stress in {INDUSTRY[1]}.")
    L.append("D = the LEAD (bellwether) borrower's financial distress; it is "
             "driven by the macro factors.")
    for y in Ys:
        L.append(f"{y} = default of downstream borrower in {INDUSTRY[io[y]]}; it "
                 f"is driven by the lead's distress D (CONTAGION) and by its "
                 f"industry factor G{io[y]}.")
    L.append("The macro factors confound D and every downstream default through "
             "the path  D <- (macro) -> (industry) -> Y. Conditional probabilities:")
    for v in g.variables:
        ps = list(g.parent_order(v))
        for combo, d in g.cpt(v).items():
            cond = ",".join(f"{p}={c}" for p, c in zip(ps, combo))
            L.append(f"P({v}=1|{cond})={d[1]:.3f}" if ps else f"P({v}=1)={d[1]:.3f}")
    if portfolio:
        L.append(f"In a stress test we SET the lead borrower to distressed, do(D=1). "
                 f"What is the PORTFOLIO default rate = the average over the "
                 f"{len(Ys)} downstream borrowers of P(Y_j=1 | do(D=1)) — the causal "
                 f"contagion effect, correctly adjusting for the systemic "
                 f"(macro+industry) confounders? Give 4 decimals.")
    else:
        L.append(f"In a stress test we SET the lead borrower to distressed, do(D=1). "
                 f"What is P({query_target}=1 | do(D=1)) — the causal contagion "
                 f"effect on this downstream borrower, correctly adjusting for the "
                 f"systemic (macro+industry) confounders? Give 4 decimals.")
    return "\n".join(L)


PROTO = "\nEnd your reply with exactly one line:\nANSWER: <number with 4 decimals>"
SYS = ("You are a quantitative model-risk expert in credit contagion. For an "
       "interventional query P(target | do(D=1)), sever D's causes and use the "
       "back-door adjustment over ALL systemic confounders (macro AND industry "
       "factors): P(target=1|do(D=1)) = sum over every macro/industry configuration "
       "c of P(target=1 | D=1, relevant-parents-induced-by-c) * P(c). Work "
       "carefully and exactly.")
_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)


def _last(content):
    m = None
    for m in _ANS.finditer(content or ""):
        pass
    return float(m.group(1)) if m else None


def ask_gpt(text):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": text + PROTO}],
                "max_completion_tokens": MAXTOK}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]
        tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0,
                "fail": str(e)[:120]}
    return {"pred": _last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if _last(c) is not None else "no ANSWER"}


# ---------------------------------------------------------------------------
# Difficulty cells: sweep number of downstream debtors. More debtors = larger
# portfolio-rate query (more marginalization for the LLM), and the back-door
# adjustment set (macro 2 + industry 2 = 4 confounders -> 2^4=16 configs) is
# the genuine combinatorial load on every single-debtor query too.
# ---------------------------------------------------------------------------
CELLS = [
    # (label, n_debtors, portfolio?, n_macro) -- sweep systemic confounders => 2^k load
    ("contagion k2 (2 macro)", 2, False, 2),
    ("contagion k3 (3 macro)", 2, False, 3),
    ("contagion k4 (4 macro)", 2, False, 4),
    ("contagion k5 (5 macro)", 2, False, 5),
    ("portfolio k4 (4 macro)", 3, True, 4),
]


def main():
    self_test()  # offline engine<->pgmpy gate BEFORE any LLM call
    do_llm = bool(os.environ.get("OPENAI_API_KEY"))
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["label"], r["i"]))
    jf = jpath.open("a"); iprg_max = 0.0
    for label, n_deb, portfolio, n_macro in CELLS:
        for i in range(N_PER_CELL):
            g, meta = cascade_scm(n_deb, BASE_SEED + 1000 * n_deb + 100 * n_macro + (500 if portfolio else 0) + i, n_macro)
            Ys = meta["Ys"]
            # engine truths: per-debtor do-effects, then portfolio = mean
            per = {}
            for y in Ys:
                per[y] = round(InterventionEngine(g).query_intervention(y, 1, {"D": 1}).value, 6)
            # IPRG: independent pgmpy recompute for every debtor
            for y in Ys:
                pg = pgmpy_do1_target(g, y)
                iprg_max = max(iprg_max, abs(pg - per[y]))
            truth = round(float(np.mean(list(per.values()))), 6) if portfolio else per["Y0"]
            qtgt = None if portfolio else "Y0"
            if (label, i) in done:
                continue
            row = {"label": label, "n_debtors": n_deb, "portfolio": portfolio,
                   "i": i, "n_nodes": meta["n_nodes"], "truth": truth,
                   "per_debtor": per,
                   "iprg": round(max(abs(pgmpy_do1_target(g, y) - per[y]) for y in Ys), 12)}
            if do_llm:
                row["gpt51"] = ask_gpt(render(g, meta, qtgt, portfolio))
            else:
                row["gpt51"] = {"pred": None, "fail": "no OPENAI_API_KEY", "skipped": True}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[{label:26} #{i:02d}] nodes={meta['n_nodes']} truth={truth:.4f} "
                  f"gpt={row['gpt51'].get('pred')} iprg={row['iprg']:.1e}", flush=True)
    jf.close()

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print(f"\nIPRG max|pgmpy-engine| = {iprg_max:.2e} -> "
          f"{'PASS' if iprg_max < 1e-6 else 'FAIL'}")
    summ = {}
    print("\n=== default-cascade contagion: gpt-5.1 vs engine (engine exact) ===")
    print(f"{'cell':>26} | {'n':>2} | {'gpt acc(±.005)':>14} | {'gpt MAE':>8} | {'fail':>6}")
    for label, n_deb, portfolio, n_macro in CELLS:
        kr = [r for r in rows if r["label"] == label]
        gp = [r for r in kr if r.get("gpt51", {}).get("pred") is not None]
        acc = float(np.mean([abs(r["gpt51"]["pred"] - r["truth"]) <= TOL for r in gp])) if gp else None
        mae = float(np.mean([abs(r["gpt51"]["pred"] - r["truth"]) for r in gp])) if gp else None
        fail = sum(1 for r in kr if r.get("gpt51", {}).get("pred") is None)
        summ[label] = {"n_debtors": n_deb, "portfolio": portfolio,
                       "gpt_acc": round(acc, 3) if acc is not None else None,
                       "gpt_mae": round(mae, 4) if mae is not None else None,
                       "fail": f"{fail}/{len(kr)}", "n_scms": len(kr)}
        print(f"{label:>26} | {n_deb:>2} | {str(summ[label]['gpt_acc']):>14} | "
              f"{str(summ[label]['gpt_mae']):>8} | {summ[label]['fail']:>6}")
    (HERE / "results.json").write_text(json.dumps(
        {"summary": summ, "iprg_max": iprg_max,
         "iprg_verdict": "PASS" if iprg_max < 1e-6 else "FAIL",
         "n_per_cell": N_PER_CELL, "tol": TOL, "rows": rows}, indent=2))
    print(f"\nresults.json + rows.jsonl written to {HERE}")


def self_test():
    """Offline gate: build a few SCMs, confirm engine returns in <2s and pgmpy
    matches every per-debtor do-effect to <1e-6. Aborts (raise) on mismatch."""
    t0 = time.time(); mx = 0.0; ncheck = 0
    for n_macro in (2, 3, 4, 5):
        for i in range(3):
            g, meta = cascade_scm(2, 99000 + 100 * n_macro + i, n_macro)
            for y in meta["Ys"]:
                eng = InterventionEngine(g).query_intervention(y, 1, {"D": 1}).value
                pg = pgmpy_do1_target(g, y)
                mx = max(mx, abs(eng - pg)); ncheck += 1
    dt = time.time() - t0
    print(f"[self_test] {ncheck} do-effects, max|engine-pgmpy|={mx:.2e}, "
          f"{dt:.2f}s total -> {'OK' if mx < 1e-6 else 'FAIL'}", flush=True)
    if mx >= 1e-6:
        raise SystemExit(f"IPRG self-test FAILED: max diff {mx:.2e} >= 1e-6; aborting before LLM.")


if __name__ == "__main__":
    main()
