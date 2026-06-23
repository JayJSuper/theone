"""MECHANISM probe: is the combinatorial cliff a fixed effective-compute-depth bound?

Hypothesis (the question an external reasoning-theory expert would raise, answered
here by experiment rather than by waiting for their reply): the cliff is the point
where exact marginalization of 2^k terms — a serial accumulation — exceeds the
serial-computation depth the model can realize within its reasoning budget. If so,
*raising the budget on the SAME model should push the cliff to a higher k* (CoT /
thinking tokens buy more effective depth), and it should *saturate* (a fixed
architecture has a ceiling no budget removes). We already have a cross-model hint —
gemini@24k held k6 while gpt-5.1@4k held k5 — but that confounds model with budget.
This isolates budget WITHIN one model (gpt-5.1) on the clean de-anchored generator.

Prediction grid: budget ∈ {2k, 8k, 32k} × k ∈ {5,6,7}. Engine exact throughout
(pgmpy IPRG per SCM). Read the cliff location (first k with accuracy < 0.5) per budget.

Run:  source ~/.theone_keys.env && .venv/bin/python experiments/cliff_mechanism/run.py
"""
from __future__ import annotations
import importlib.util, json, os, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

HERE = Path(__file__).parent


def _load(name, rel):
    s = importlib.util.spec_from_file_location(name, HERE.parent / rel)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

DA = _load("deanchor", "deanchor_cliff/run.py")     # k_graph_skewed + R
CV = _load("crossval", "complexity_axis/cross_validate.py")  # to_pgmpy
R = DA.R
N = 6
TOL = 0.005
BUDGETS = [2048, 8192, 32768]
KS = [5, 6, 7]


def pgmpy_do1(g, k):
    """Independent do(X=1) via graph surgery in pgmpy (IPRG gate)."""
    m = CV.to_pgmpy(g, k).copy()
    for p in list(m.predecessors("X")):
        m.remove_edge(p, "X")
    m.remove_cpds(m.get_cpds("X"))
    m.add_cpds(TabularCPD("X", 2, [[0.0], [1.0]], state_names={"X": [0, 1]}))
    m.check_model()
    return float(VariableElimination(m).query(["Y"], show_progress=False).values[1])


def ask(text, budget, tmo=300):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": R.SYS},
                {"role": "user", "content": text + R.PROTO}],
                "max_completion_tokens": budget}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=tmo) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]
        tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:90]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["budget"], r["k"], r["i"]))
    jf = jpath.open("a"); iprg_max = 0.0
    for budget in BUDGETS:
        for k in KS:
            for i in range(N):
                g = DA.k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)
                truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
                pg = pgmpy_do1(g, k); iprg_max = max(iprg_max, abs(pg - truth))
                if (budget, k, i) in done:
                    continue
                res = ask(R.render(g, k), budget)
                row = {"budget": budget, "k": k, "i": i, "truth": truth,
                       "pgmpy": round(pg, 6), "iprg": round(abs(pg - truth), 12), "gpt": res}
                jf.write(json.dumps(row) + "\n"); jf.flush()
                print(f"[budget={budget:>5} k={k} #{i}] truth={truth:.3f} pgmpy={pg:.3f} "
                      f"gpt={res.get('pred')} tok={res.get('tokens')} ({res.get('latency')}s)", flush=True)
    jf.close()

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print(f"\nIPRG max|pgmpy-engine| = {iprg_max:.2e} -> {'PASS' if iprg_max < 1e-6 else 'FAIL'}")
    print("\n=== cliff location vs reasoning budget (gpt-5.1, de-anchored) ===")
    print(f"{'budget':>8} | " + " | ".join(f"k={k} acc" for k in KS) + " | cliff (first k<0.5)")
    summ = {}
    for budget in BUDGETS:
        accs = {}
        for k in KS:
            kr = [r for r in rows if r["budget"] == budget and r["k"] == k]
            gp = [r for r in kr if r["gpt"]["pred"] is not None]
            accs[k] = float(np.mean([abs(r["gpt"]["pred"] - r["truth"]) <= TOL for r in gp])) if gp else 0.0
        cliff = next((k for k in KS if accs[k] < 0.5), None)
        summ[budget] = {"acc": accs, "cliff_k": cliff}
        print(f"{budget:>8} | " + " | ".join(f"{accs[k]:.2f}  " for k in KS) + f" | {cliff}")
    (HERE / "results.json").write_text(json.dumps(summ, indent=2))
    print("\nIf cliff_k RISES with budget -> cliff is an effective-compute-depth bound "
          "(CoT buys depth); if FLAT -> budget is not the binding constraint here.")


if __name__ == "__main__":
    main()
