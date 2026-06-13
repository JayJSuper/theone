"""DIAGNOSTIC (k=1b): same SCMs/SEED as run.py (k=1), but the distractor P(X|U)
is phrased DISMISSIVELY (no numbers) instead of as concrete percentages.

Purpose: isolate WHY k=1 raw drifted (MAE 0.065) while k=2 raw was exact. The
two runs differed in distractor phrasing. If k=1b raw -> exact, the original k=1
drift was DISTRACTOR CONTAMINATION (gpt-5.1 lured by irrelevant stated numbers),
NOT marginalization-arithmetic load. That would force a correction to AM-015.

Run: source ~/.theone_keys.env && python experiments/wildtext_numeric/run_k1b.py
"""
from __future__ import annotations
import importlib.util, json, re
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
_s = importlib.util.spec_from_file_location("k1", HERE / "run.py")
k1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(k1)

def narrate_dismissive(X, Y, U, consistent, pu, px1, px0, v):
    """Identical to k1.narrate but the distractor carries NO numbers."""
    arrow = (f"In this population {X} causally drives {Y} (it occurs earlier and "
             f"changes {Y} downstream).")
    cn = ("This is consistent with common sense." if consistent else
          "NOTE: this REVERSES real-world common sense — treat the stated direction "
          "as ground truth for this population anyway.")
    return (
        f"A cohort registry studied whether {X} affects {Y}, recording {U} as a "
        f"common cause of both. {arrow} {cn} "
        f"{U} was present in {k1.pct(pu)}% of the cohort. "
        f"Among participants WITH {U}: {k1.pct(v[(1,1)])}% of those with {X} showed "
        f"{Y}, versus {k1.pct(v[(0,1)])}% of those without {X}. "
        f"Among participants WITHOUT {U}: {k1.pct(v[(1,0)])}% of those with {X} showed "
        f"{Y}, versus {k1.pct(v[(0,0)])}% of those without {X}. "
        f"(The treatment's own prevalence varied with {U} but is not the question here.)")

def main():
    jpath = HERE / "rows_k1b.jsonl"
    done = {json.loads(l)["i"] for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jf = jpath.open("a")
    for i in range(k1.N_ITEMS):
        X, Y, U, consistent = k1.WORDS[i % len(k1.WORDS)]
        g, pu, px1, px0, v, do, obs = k1.make_scm(np.random.default_rng(k1.SEED + 1 + i))
        if i in done:
            continue
        narr = narrate_dismissive(X, Y, U, consistent, pu, px1, px0, v)
        sys = "You are a careful causal-inference expert."
        try:
            txt = k1._openai("gpt-5.1", sys, narr + "\n\n" + k1.SOLVE.format(X=X, Y=Y))
            mm = list(k1._ANS.finditer(txt)); raw = float(mm[-1].group(1)) if mm else None
        except Exception:
            raw = None
        row = {"i": i, "consistent": consistent, "truth_do": do, "obs": obs,
               "gap": round(abs(do-obs), 4), "raw": raw, "narrative": narr}
        jf.write(json.dumps(row) + "\n"); jf.flush()
        re_ = None if raw is None else round(abs(raw-do), 3)
        print(f"[{i:02d}] {'CN' if consistent else 'CT'} truth={do:.3f} obs={obs:.3f} raw={raw} err={re_}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines()]
    ok = [r for r in rows if r.get("raw") is not None]
    strict = round(sum(1 for r in ok if abs(r["raw"]-r["truth_do"]) <= k1.TOL)/len(ok), 3)
    mae = round(float(np.mean([abs(r["raw"]-r["truth_do"]) for r in ok])), 4)
    mx = round(max(abs(r["raw"]-r["truth_do"]) for r in ok), 4)
    print("\n===== k=1b (dismissive distractor) =====")
    print(json.dumps({"n": len(ok), "raw_acc_strict": strict, "raw_mae": mae, "raw_max": mx,
                      "COMPARE_original_k1": {"raw_acc_strict": 0.042, "raw_mae": 0.0649, "raw_max": 0.211}}, indent=2))

if __name__ == "__main__":
    main()
