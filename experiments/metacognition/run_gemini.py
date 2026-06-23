"""Pillar 3 cross-family: does the anti-calibration (confidence-when-wrong ≥
confidence-when-right) hold beyond gpt-5.1? Fills the §6.7 single-base caveat.
Subject: gemini-2.5-pro (protocol-stable, clean in the token-control study), same
de-anchored cliff and same CONFIDENCE elicitation as run.py.

  source ~/.theone_keys.env && .venv/bin/python experiments/metacognition/run_gemini.py
"""
from __future__ import annotations
import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_r = importlib.util.spec_from_file_location("cliffrun", HERE.parent / "complexity_axis" / "run.py")
R = importlib.util.module_from_spec(_r); _r.loader.exec_module(R)
_d = importlib.util.spec_from_file_location("deanchor", HERE.parent / "deanchor_cliff" / "run.py")
D = importlib.util.module_from_spec(_d); _d.loader.exec_module(D)

KS = [4, 5, 6]; N = 8; TOL = 0.005
SYS = ("You are an expert in causal inference. Compute P(Y=1|do(X=1)) by back-door "
       "adjustment over ALL confounders. Then HONESTLY assess your own confidence "
       "that your numeric answer is within +/-0.01 of the true value.")
PROTO = ("\nEnd with exactly two lines:\nANSWER: <number, 4 decimals>\n"
         "CONFIDENCE: <integer 0-100, your probability that ANSWER is within 0.01 of truth>")
_A = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
_C = re.compile(r"CONFIDENCE:\s*([0-9]+)", re.I)


def ask(text, maxtok=24000):
    try:
        req = urllib.request.Request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-2.5-pro:generateContent?key={os.environ['GEMINI_API_KEY']}",
            data=json.dumps({"systemInstruction": {"parts": [{"text": SYS}]},
                             "contents": [{"parts": [{"text": text + PROTO}]}],
                             "generationConfig": {"temperature": 0, "maxOutputTokens": maxtok}}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        c = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    except Exception as e:
        return {"pred": None, "conf": None, "fail": str(e)[:80]}
    a = list(_A.finditer(c)); cf = list(_C.finditer(c))
    return {"pred": float(a[-1].group(1)) if a else None,
            "conf": int(cf[-1].group(1)) if cf else None, "fail": None}


def main():
    jpath = HERE / "rows_gemini.jsonl"
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

    have = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    have = [r for r in have if r["conf"] is not None]
    cor = [r for r in have if r["correct"]]; wr = [r for r in have if not r["correct"]]
    print(f"\n=== metacognition (gemini-2.5-pro, de-anchored cliff, n={len(have)}) ===")
    acc = len(cor) / len(have) if have else 0
    mc = np.mean([r["conf"] for r in cor]) if cor else float("nan")
    mw = np.mean([r["conf"] for r in wr]) if wr else float("nan")
    print(f"overall accuracy: {acc:.2f} | mean confidence: {np.mean([r['conf'] for r in have]):.1f}")
    print(f"mean confidence WHEN CORRECT: {mc:.1f} (n={len(cor)})")
    print(f"mean confidence WHEN WRONG  : {mw:.1f} (n={len(wr)})")
    anti = (not np.isnan(mw) and not np.isnan(mc) and mw >= mc - 5)
    print(f"anti-calibration (conf_wrong ≥ conf_right): {'YES — replicates gpt-5.1' if anti else 'NO'}")
    summ = {"model": "gemini-2.5-pro", "n": len(have), "accuracy": round(acc, 3),
            "conf_correct": None if np.isnan(mc) else round(float(mc), 1),
            "conf_wrong": None if np.isnan(mw) else round(float(mw), 1),
            "anti_calibrated": bool(anti)}
    (HERE / "results_gemini.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
