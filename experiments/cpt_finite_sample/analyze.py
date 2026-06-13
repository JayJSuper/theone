"""Analyze the CPT finite-sample ecological-validity experiment.

Per (k, n) computes:
  engine_err   = |engine.do(EST) - true_do|        # irreducible estimation floor
  gpt/ds_err   = |llm - true_do|                    # floor + reasoning
  reasoning    = |llm - engine.do(EST)|             # reasoning error, isolated from estimation
  llm accuracy = fraction within TOL of true_do
The headline question: does the engine advantage (llm_err - engine_err) survive
CPT noise, and does the reasoning collapse persist at high k under estimated CPTs?

Run: python experiments/cpt_finite_sample/analyze.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TOL = 0.005


def main():
    rows = [json.loads(l) for l in (HERE / "rows.jsonl").read_text().splitlines() if l.strip()]
    ks = sorted({r["k"] for r in rows}); ns = sorted({r["n"] for r in rows})
    iprg = [r["iprg_pgmpy_diff"] for r in rows if "iprg_pgmpy_diff" in r]
    print(f"rows={len(rows)}  IPRG max|pgmpy-engine|={max(iprg):.2e} ({len(iprg)} checked) "
          f"-> {'PASS' if max(iprg)<1e-6 else 'FAIL'}\n")

    print("ENGINE estimation floor — mean |engine.do(EST) - true_do| (no API, full arm):")
    print(f"  {'k':>3} | " + " ".join(f"n={n:<6}" for n in ns))
    floor = {}
    for k in ks:
        cells = []
        for n in ns:
            errs = [r["engine_err"] for r in rows if r["k"] == k and r["n"] == n]
            floor[(k, n)] = np.mean(errs) if errs else float("nan")
            cells.append(f"{floor[(k,n)]:.4f}".ljust(8))
        print(f"  {k:>3} | " + " ".join(cells))

    for model in ("gpt51", "deepseek"):
        print(f"\n{model.upper()} on the SAME estimated CPT:")
        print(f"  {'k':>3} {'n':>5} | {'llm_err':>8} {'eng_floor':>9} {'reasoning':>9} {'llm_acc':>8} {'eng>llm?':>9}")
        for k in ks:
            for n in ns:
                sub = [r for r in rows if r["k"] == k and r["n"] == n and model in r and r[model].get("pred") is not None]
                if not sub:
                    continue
                lerr = np.mean([abs(r[model]["pred"] - r["true_do"]) for r in sub])
                reas = np.mean([abs(r[model]["pred"] - r["engine_est"]) for r in sub])
                acc = np.mean([1 if abs(r[model]["pred"] - r["true_do"]) <= TOL else 0 for r in sub])
                fl = floor[(k, n)]
                fails = sum(1 for r in [x for x in rows if x["k"]==k and x["n"]==n and model in x] if x.get(model,{}).get("pred") is None) if False else 0
                adv = "ENGINE" if lerr > fl + 0.005 else "tie"
                print(f"  {k:>3} {n:>5} | {lerr:>8.4f} {fl:>9.4f} {reas:>9.4f} {acc:>8.2f} {adv:>9}")
    # protocol failures (pred None) per model at high k
    print("\nprotocol failures (no parseable answer) by k:")
    for model in ("gpt51", "deepseek"):
        per = {k: sum(1 for r in rows if r["k"]==k and model in r and r[model].get("pred") is None) for k in ks}
        tot = {k: sum(1 for r in rows if r["k"]==k and model in r) for k in ks}
        print(f"  {model}: " + " ".join(f"k{k}={per[k]}/{tot[k]}" for k in ks))


if __name__ == "__main__":
    main()
