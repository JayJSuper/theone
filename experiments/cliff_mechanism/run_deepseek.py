"""MECHANISM probe, cross-model: does budget move the cliff for a WEAKER model?

The gpt-5.1 run (run.py) found the cliff is NOT budget-movable within that model — a
capability/architecture wall. But that is one model. A weaker model (deepseek-v4-flash,
cliff at k=4, and known to protocol-fail at high k) might be the case where budget IS
the binding constraint: maybe small budgets truncate its reasoning and a larger budget
lets it reach a higher k. We do not presume the gpt result generalizes — we test it.

  source ~/.theone_keys.env && .venv/bin/python experiments/cliff_mechanism/run_deepseek.py
"""
from __future__ import annotations
import importlib.util, json, time
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine
from theone.llm import DeepSeekClient

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("mech", HERE / "run.py")
M = importlib.util.module_from_spec(_s); _s.loader.exec_module(M)
DA, R, pgmpy_do1 = M.DA, M.R, M.pgmpy_do1
N = 6
TOL = 0.005
BUDGETS = [2048, 8192, 32768]
KS = [3, 4, 5]          # deepseek cliff is k=4, so probe around it


def ask_ds(text, budget):
    t0 = time.time()
    try:
        out = DeepSeekClient(timeout=240).chat(
            [{"role": "system", "content": R.SYS},
             {"role": "user", "content": text + R.PROTO}],
            max_tokens=budget, temperature=0.0)
        c, tok = out["content"], out["usage"].get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:90]}
    return {"pred": R._last(c), "latency": round(time.time() - t0, 1), "tokens": tok,
            "fail": None if R._last(c) is not None else "no ANSWER"}


def main():
    jpath = HERE / "rows_deepseek.jsonl"
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
                res = ask_ds(R.render(g, k), budget)
                row = {"budget": budget, "k": k, "i": i, "truth": truth,
                       "pgmpy": round(pg, 6), "iprg": round(abs(pg - truth), 12), "ds": res}
                jf.write(json.dumps(row) + "\n"); jf.flush()
                print(f"[budget={budget:>5} k={k} #{i}] truth={truth:.3f} "
                      f"ds={res.get('pred')} tok={res.get('tokens')} fail={res.get('fail') is not None}", flush=True)
    jf.close()

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    print(f"\nIPRG max|pgmpy-engine| = {iprg_max:.2e} -> {'PASS' if iprg_max < 1e-6 else 'FAIL'}")
    print("\n=== deepseek-v4-flash: cliff location + protocol-fail vs budget ===")
    print(f"{'budget':>8} | " + " | ".join(f"k={k}" for k in KS) + " | cliff(first acc<0.5)")
    summ = {}
    for budget in BUDGETS:
        accs, fails = {}, {}
        for k in KS:
            kr = [r for r in rows if r["budget"] == budget and r["k"] == k]
            gp = [r for r in kr if r["ds"]["pred"] is not None]
            accs[k] = float(np.mean([abs(r["ds"]["pred"] - r["truth"]) <= TOL for r in gp])) if gp else 0.0
            fails[k] = len(kr) - len(gp)
        cliff = next((k for k in KS if accs[k] < 0.5), None)
        summ[budget] = {"acc": accs, "protocol_fails": fails, "cliff_k": cliff}
        print(f"{budget:>8} | " + " | ".join(f"{accs[k]:.2f}(f{fails[k]})" for k in KS) + f" | {cliff}")
    (HERE / "results_deepseek.json").write_text(json.dumps(summ, indent=2))
    print("\nCompare to gpt-5.1 (cliff fixed at k5, budget-immune). If deepseek's cliff "
          "ALSO stays put -> capability-wall generalizes; if it MOVES with budget -> "
          "weak models are budget-bound, strong models architecture-bound (finer picture).")


if __name__ == "__main__":
    main()
