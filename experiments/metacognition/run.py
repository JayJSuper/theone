"""Pillar 3 — calibrated metacognition: does the LLM KNOW when it is wrong?
The confident-wrong cliff finding (precise answers, no hedging, even P>1) is a
metacognition failure. Here we elicit gpt-5.1's own CONFIDENCE alongside its
answer on the de-anchored cliff (k=4,5,6) and measure whether confidence tracks
correctness. If confidence-when-wrong ~= confidence-when-right, the model has no
usable metacognition — which is exactly what a verifiable cognitive credential
replaces (the credential's correctness is checkable, not self-reported).

Run: source ~/.theone_keys.env && python experiments/metacognition/run.py
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

KS = [4, 5, 6]; N = 20; TOL = 0.005
SYS = ("You are an expert in causal inference. Compute P(Y=1|do(X=1)) by back-door "
       "adjustment over ALL confounders. Then HONESTLY assess your own confidence "
       "that your numeric answer is within +/-0.01 of the true value.")
PROTO = ("\nEnd with exactly two lines:\nANSWER: <number, 4 decimals>\n"
         "CONFIDENCE: <integer 0-100, your probability that ANSWER is within 0.01 of truth>")
_A = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
_C = re.compile(r"CONFIDENCE:\s*([0-9]+)", re.I)


def ask(text, maxtok=4096):
    t0 = time.time()
    try:
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": SYS}, {"role": "user", "content": text + PROTO}],
                "max_completion_tokens": maxtok}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            c = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return {"pred": None, "conf": None, "fail": str(e)[:80]}
    a = list(_A.finditer(c)); cf = list(_C.finditer(c))
    return {"pred": float(a[-1].group(1)) if a else None,
            "conf": int(cf[-1].group(1)) if cf else None, "fail": None}


def main():
    jpath = HERE / "rows.jsonl"
    done = {(json.loads(l)["k"], json.loads(l)["i"]) for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jf = jpath.open("a")
    for k in KS:
        for i in range(N):
            if (k, i) in done: continue
            g = D.k_graph_skewed(k, R.BASE_SEED + 1000 * k + i)
            truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
            o = ask(R.render(g, k))
            correct = o["pred"] is not None and abs(o["pred"] - truth) <= TOL
            row = {"k": k, "i": i, "truth": truth, "pred": o["pred"], "conf": o["conf"], "correct": bool(correct)}
            jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"[k{k}-{i:02d}] truth={truth:.3f} pred={o['pred']} conf={o['conf']} correct={correct}", flush=True)
    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    have = [r for r in rows if r["conf"] is not None and r["pred"] is not None]
    cor = [r for r in have if r["correct"]]; wr = [r for r in have if not r["correct"]]
    print(f"\n=== metacognition (gpt-5.1, de-anchored cliff, n={len(have)}) ===")
    print(f"overall accuracy: {len(cor)}/{len(have)} = {len(cor)/len(have):.2f}")
    print(f"mean confidence WHEN CORRECT: {np.mean([r['conf'] for r in cor]):.1f}" if cor else "no correct")
    print(f"mean confidence WHEN WRONG  : {np.mean([r['conf'] for r in wr]):.1f}" if wr else "no wrong")
    # per-k: accuracy vs mean confidence (the gap = overconfidence)
    print(f"\n{'k':>2} {'acc':>5} {'mean_conf':>10} {'overconfidence(conf-acc)':>24}")
    summ = {}
    for k in KS:
        kk = [r for r in have if r["k"] == k]
        acc = np.mean([r["correct"] for r in kk]); mc = np.mean([r["conf"] for r in kk]) / 100
        summ[k] = {"acc": round(float(acc), 3), "mean_conf": round(float(mc), 3), "overconf": round(float(mc - acc), 3)}
        print(f"{k:>2} {acc:>5.2f} {mc:>10.2f} {mc-acc:>+24.2f}")
    (HERE / "results.json").write_text(json.dumps({"summary": summ,
        "conf_when_correct": float(np.mean([r['conf'] for r in cor])) if cor else None,
        "conf_when_wrong": float(np.mean([r['conf'] for r in wr])) if wr else None,
        "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
