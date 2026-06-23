"""Pillar 3, finally forcing the wrong samples: deepseek-v4-flash @32k at k=8 (256
configs) — beyond what even 32k can marginalize, so it MUST err. Then its confidence-
when-wrong is directly measurable, closing the question left open at k≤7 (where it was
all-correct). If wrong-but-still-high-confidence → confidence is flat-high, decoupled
from completion (anti-calibration is the same phenomenon, now shown on a 3rd family
even at large budget). If wrong-and-lower → it knows when it cannot finish.

  source ~/.theone_keys.env && .venv/bin/python experiments/metacognition/run_deepseek_k8.py
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

KS = [8]; N = 12; TOL = 0.005; BUDGET = 32768
SYS = ("You are an expert in causal inference. Compute P(Y=1|do(X=1)) by back-door "
       "adjustment over ALL confounders. Then HONESTLY assess your own confidence "
       "that your numeric answer is within +/-0.01 of the true value.")
PROTO = ("\nEnd with exactly two lines:\nANSWER: <number, 4 decimals>\n"
         "CONFIDENCE: <integer 0-100, your probability that ANSWER is within 0.01 of truth>")
_A = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
_C = re.compile(r"CONFIDENCE:\s*([0-9]+)", re.I)


def ask(text):
    try:
        out = DeepSeekClient(timeout=300).chat(
            [{"role": "system", "content": SYS}, {"role": "user", "content": text + PROTO}],
            max_tokens=BUDGET, temperature=0.0)
        c = out["content"] or ""
    except Exception as e:
        return {"pred": None, "conf": None, "fail": str(e)[:80]}
    a = list(_A.finditer(c)); cf = list(_C.finditer(c))
    return {"pred": float(a[-1].group(1)) if a else None,
            "conf": int(cf[-1].group(1)) if cf else None, "fail": None}


def main():
    jpath = HERE / "rows_deepseek_k8.jsonl"
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
    nfail = sum(1 for l in jpath.read_text().splitlines() if l.strip() and json.loads(l)["conf"] is None)
    acc = len(cor) / len(have) if have else 0
    print(f"\n=== deepseek-v4-flash @32k, k=8 (256 configs, out of reach) n={len(have)} fail={nfail} ===")
    print(f"accuracy={acc:.2f}")
    mc = np.mean([r["conf"] for r in cor]) if cor else float("nan")
    mw = np.mean([r["conf"] for r in wr]) if wr else float("nan")
    print(f"conf WHEN CORRECT={mc:.1f} (n={len(cor)}) | WHEN WRONG={mw:.1f} (n={len(wr)})")
    if not np.isnan(mw):
        flat = (np.isnan(mc) or mw >= mc - 8)
        verdict = ("FLAT-HIGH even when wrong at large budget — confidence decoupled "
                   "from completion confirmed on deepseek too (3rd family)" if flat and mw >= 70
                   else "lower when wrong — deepseek knows it cannot finish")
        print(f"VERDICT: {verdict}")
        (HERE / "results_deepseek_k8.json").write_text(json.dumps(
            {"model": "deepseek-v4-flash", "budget": BUDGET, "k": 8, "n": len(have),
             "accuracy": round(acc, 3), "conf_correct": None if np.isnan(mc) else round(float(mc), 1),
             "conf_wrong": round(float(mw), 1), "n_wrong": len(wr), "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
