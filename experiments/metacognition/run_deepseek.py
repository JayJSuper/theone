"""Pillar 3, the developing question: is anti-calibration a fixed model trait, or a
by-product of 'cannot compute'? deepseek-v4-flash, given a LARGE budget (16k), can
actually compute exact do-effects through k≈4 (NOTE-026). So: when it is genuinely
*right*, is its confidence higher than when *wrong* (= real calibration), or still
flat-high (= anti-calibration is intrinsic)? This could refine the cross-family
anti-calibration claim — exactly the spirit that no conclusion is final.

  source ~/.theone_keys.env && .venv/bin/python experiments/metacognition/run_deepseek.py
"""
from __future__ import annotations
import importlib.util, json, re
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine
from theone.llm import DeepSeekClient

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_d = importlib.util.spec_from_file_location("deanchor", HERE.parent / "deanchor_cliff" / "run.py")
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)

KS = [3, 4, 5]; N = 8; TOL = 0.005; BUDGET = 16384
SYS = ("You are an expert in causal inference. Compute P(Y=1|do(X=1)) by back-door "
       "adjustment over ALL confounders. Then HONESTLY assess your own confidence "
       "that your numeric answer is within +/-0.01 of the true value.")
PROTO = ("\nEnd with exactly two lines:\nANSWER: <number, 4 decimals>\n"
         "CONFIDENCE: <integer 0-100, your probability that ANSWER is within 0.01 of truth>")
_A = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
_C = re.compile(r"CONFIDENCE:\s*([0-9]+)", re.I)


def ask(text):
    try:
        out = DeepSeekClient(timeout=240).chat(
            [{"role": "system", "content": SYS}, {"role": "user", "content": text + PROTO}],
            max_tokens=BUDGET, temperature=0.0)
        c = out["content"] or ""
    except Exception as e:
        return {"pred": None, "conf": None, "fail": str(e)[:80]}
    a = list(_A.finditer(c)); cf = list(_C.finditer(c))
    return {"pred": float(a[-1].group(1)) if a else None,
            "conf": int(cf[-1].group(1)) if cf else None, "fail": None}


def main():
    jpath = HERE / "rows_deepseek.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                r = json.loads(l); done.add((r["k"], r["i"]))
    jf = jpath.open("a")
    for k in KS:
        for i in range(N):
            if (k, i) in done:
                continue
            g = D.k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
            o = ask(R.render(g, k))
            correct = (o["pred"] is not None and abs(o["pred"] - truth) <= TOL)
            row = {"k": k, "i": i, "truth": truth, "pred": o["pred"], "conf": o["conf"],
                   "correct": bool(correct), "fail": o["fail"]}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k} #{i}] truth={truth:.3f} pred={o['pred']} conf={o['conf']} correct={correct}", flush=True)
    jf.close()

    have = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip() and json.loads(l)["conf"] is not None]
    cor = [r for r in have if r["correct"]]; wr = [r for r in have if not r["correct"]]
    print(f"\n=== metacognition (deepseek-v4-flash, BUDGET={BUDGET}, n={len(have)}) ===")
    for k in KS:
        kr = [r for r in have if r["k"] == k]
        if kr:
            print(f"  k={k}: acc={np.mean([r['correct'] for r in kr]):.2f} mean_conf={np.mean([r['conf'] for r in kr]):.0f} (n={len(kr)})")
    mc = np.mean([r["conf"] for r in cor]) if cor else float("nan")
    mw = np.mean([r["conf"] for r in wr]) if wr else float("nan")
    print(f"conf WHEN CORRECT={mc:.1f} (n={len(cor)}) | WHEN WRONG={mw:.1f} (n={len(wr)})")
    if not np.isnan(mc) and not np.isnan(mw):
        if mc > mw + 5:
            verdict = "CALIBRATED — refines the claim: anti-calibration is a by-product of cannot-compute, not intrinsic"
        elif mw >= mc - 5:
            verdict = "ANTI-CALIBRATED — anti-calibration is intrinsic, holds even when it can compute"
        else:
            verdict = "ambiguous"
        print(f"VERDICT: {verdict}")
        (HERE / "results_deepseek.json").write_text(json.dumps(
            {"model": "deepseek-v4-flash", "budget": BUDGET, "n": len(have),
             "conf_correct": round(float(mc), 1), "conf_wrong": round(float(mw), 1),
             "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
