"""Cross-base confirmation of 'scaffolding is harmful at scale' (paper §5.1).
The scale-axis finding (raw A > scaffold B) was on deepseek only. Here we test the
SAME two system prompts (SYS_A minimal vs SYS_B causal-inference scaffold) on
gpt-5.1, at the large tier (12-node DAGs) where the effect was strongest
(deepseek L: A 0.613 vs B 0.447). Same SCMs for A and B (paired).

Run: source ~/.theone_keys.env && python experiments/scaffold_crossbase/run.py
"""
from __future__ import annotations
import importlib.util, json, os, re, time, urllib.request
from pathlib import Path
import numpy as np
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("scalerun", HERE.parent / "baseline_b_v1_scale" / "run.py")
S = importlib.util.module_from_spec(_s); _s.loader.exec_module(S)

N = 30
N_NODES = 12          # large tier
SEED = 20260620       # fresh seed (not the frozen formal run)
TOL = 0.005


def ask_gpt(system, text, maxtok=4096):
    t0 = time.time()
    try:
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({"model": "gpt-5.1", "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text + S.PROTO}],
                "max_completion_tokens": maxtok}).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
        c = out["choices"][0]["message"]["content"]; tok = out.get("usage", {}).get("total_tokens", 0)
    except Exception as e:
        return {"pred": None, "latency": round(time.time() - t0, 1), "tokens": 0, "fail": str(e)[:80]}
    m = None
    for m in re.finditer(r"ANSWER:\s*([0-9]*\.?[0-9]+)", c or "", re.I):
        pass
    return {"pred": float(m.group(1)) if m else None, "latency": round(time.time() - t0, 1),
            "tokens": tok, "fail": None if m else "no ANSWER"}


def main():
    jpath = HERE / "rows.jsonl"
    done = set()
    if jpath.exists():
        for l in jpath.read_text().splitlines():
            if l.strip():
                done.add(json.loads(l)["i"])
    jf = jpath.open("a")
    rng = np.random.default_rng(SEED)
    for i in range(N):
        d = S.gen_dag(N_NODES, rng)                  # advance rng even if done
        if i in done:
            continue
        g = S.build_graph(d)
        truth = round(InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value, 6)
        text = S.render_text(d)
        a = ask_gpt(S.SYS_A, text); b = ask_gpt(S.SYS_B, text)
        row = {"i": i, "truth": truth, "A": a, "B": b}
        jf.write(json.dumps(row) + "\n"); jf.flush()
        print(f"[{i:02d}] truth={truth:.3f} A={a.get('pred')}({a.get('latency')}s) B={b.get('pred')}({b.get('latency')}s)", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines() if l.strip()]
    def summ(arm):
        ok = [r for r in rows if r[arm]["pred"] is not None]
        acc = np.mean([abs(r[arm]["pred"] - r["truth"]) <= TOL for r in ok]) if ok else None
        fail = sum(1 for r in rows if r[arm]["pred"] is None)
        lat = np.mean([r[arm]["latency"] for r in rows])
        tok = np.mean([r[arm]["tokens"] for r in rows])
        return acc, fail, lat, tok
    print(f"\n=== scaffold cross-base on gpt-5.1, large tier (n={len(rows)}) ===")
    for arm, name in (("A", "raw (SYS_A)"), ("B", "scaffold (SYS_B)")):
        acc, fail, lat, tok = summ(arm)
        print(f"  {name:18}: acc={round(acc,3) if acc is not None else None} protocol_fail={fail}/{len(rows)} lat={lat:.1f}s tok={tok:.0f}")
    print("  (deepseek reference, large tier: A 0.613 vs B 0.447 -> scaffold harmful)")
    (HERE / "results.json").write_text(json.dumps({"n": len(rows),
        "A": summ("A"), "B": summ("B"), "rows": rows}, indent=2, default=str))


if __name__ == "__main__":
    main()
