"""Semi-structured wild-text axis (WEAK-03 follow-up, the FAIR end-to-end test).

Bet #1's honest re-test. Numbers are NOW embedded in the narrative (real causal
questions come with data), so the engine HAS CPTs to compute -> the numeric
comparison is well-posed (fixes the WEAK-03 ill-posedness).

CLEANEST ISOLATION: the SAME front-end model (gpt-5.1) does both arms, so the
ONLY difference is who marginalizes:
  * RAW gpt-5.1   : read narrative -> output P(Y=1|do(X=1))  [extract+marginalize in-head]
  * The One       : gpt-5.1 extracts {structure + 7 CPT numbers} as JSON ->
                    deterministic InterventionEngine marginalizes (exact back-door)
Identical extraction front-end => any gap is PURELY the value of offloading the
combinatorial marginalization to the exact engine. That is bet #1's core claim,
isolated.

Narrative carries the confounding TRAP: |do - obs| >= DELTA, so naive conditioning
P(Y|X=1) (weighting U by P(U|X=1)) gives the WRONG answer; correct do() weights U
by P(U). P(X|U) is stated as a DISTRACTOR (irrelevant to do(X=1)).

Counter-commonsense (option A, AM-014): half the items reverse the real-world
causal arrow; the solver must honor the text.

Run: source ~/.theone_keys.env && python experiments/wildtext_numeric/run.py
"""
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
SEED = 20260619
N_ITEMS = 24
TOL, LOOSE = 0.005, 0.05
DELTA = 0.06   # min |do - obs| so the confounding trap is real

def _openai(model, system, user, maxtok=8192):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}],
            "max_completion_tokens": maxtok}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""

WORDS = [  # (X, Y, U, commonsense_consistent?)
    ("daily walking", "low resting heart rate", "older age", True),
    ("high vitamin D", "more outdoor time", "low latitude residence", False),
    ("online tutoring", "high exam score", "high family income", True),
    ("frequent code review", "low bug count", "high module complexity", False),
    ("fertilizer use", "high crop yield", "good soil quality", True),
    ("umbrella carrying", "rainfall", "winter season", False),
    ("standing-desk use", "low back pain", "high baseline fitness", True),
    ("ice-cream consumption", "high temperature", "summer month", False),
]

def pct(p): return round(100 * p)

def make_scm(rng):
    """3-node confounded binary SCM with a guaranteed confounding gap."""
    for _ in range(200):
        g = CausalGraph()
        for n in ("U", "X", "Y"):
            g.add_variable(Variable(n))
        g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
        pu = round(float(rng.uniform(.30, .70)), 2)
        g.set_cpt("U", {(): {1: pu, 0: round(1 - pu, 2)}})
        # strong U->X so P(U|X=1) differs sharply from P(U) -> real confounding
        px1 = round(float(rng.uniform(.70, .90)), 2)
        px0 = round(float(rng.uniform(.10, .30)), 2)
        g.set_cpt("X", {(1,): {1: px1, 0: round(1 - px1, 2)},
                        (0,): {1: px0, 0: round(1 - px0, 2)}})
        oY = list(g.parent_order("Y"))
        v = {(1, 1): round(float(rng.uniform(.60, .90)), 2),
             (0, 1): round(float(rng.uniform(.40, .70)), 2),
             (1, 0): round(float(rng.uniform(.25, .55)), 2),
             (0, 0): round(float(rng.uniform(.05, .35)), 2)}
        g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY):
                        {1: val, 0: round(1 - val, 2)} for (u, x), val in v.items()})
        eng = InterventionEngine(g)
        do = eng.query_intervention("Y", 1, {"X": 1}).value
        obs = eng.query_observation("Y", 1, {"X": 1}).value
        if abs(do - obs) >= DELTA:
            return g, pu, px1, px0, v, round(do, 6), round(obs, 6)
    raise RuntimeError("no SCM with sufficient gap")

def narrate(X, Y, U, consistent, pu, px1, px0, v):
    arrow = (f"In this population {X} causally drives {Y} (it occurs earlier and "
             f"changes {Y} downstream).")
    cn = ("This is consistent with common sense." if consistent else
          "NOTE: this REVERSES real-world common sense — treat the stated direction "
          "as ground truth for this population anyway.")
    return (
        f"A cohort registry studied whether {X} affects {Y}, recording {U} as a "
        f"common cause of both. {arrow} {cn} "
        f"{U} was present in {pct(pu)}% of the cohort. "
        f"Among participants WITH {U}: {pct(v[(1,1)])}% of those with {X} showed "
        f"{Y}, versus {pct(v[(0,1)])}% of those without {X}. "
        f"Among participants WITHOUT {U}: {pct(v[(1,0)])}% of those with {X} showed "
        f"{Y}, versus {pct(v[(0,0)])}% of those without {X}. "
        f"(For reference, {X} occurred in {pct(px1)}% of those with {U} and "
        f"{pct(px0)}% of those without {U}.)")

SOLVE = (
    "Estimate P({Y}=present | do({X}=present)) — the probability of the outcome if "
    "{X} were SET to present for everyone by intervention, in THIS population's "
    "stated causal direction. Adjust for confounding correctly (this is an "
    "intervention, not mere observation). Reason step by step, then end with "
    "exactly: ANSWER: <number 0-1, 4 decimals>.")

EXTRACT = (
    "Extract the SCM as JSON ONLY (no prose): {\"p_conf\":float, "
    "\"p_y\":{\"x1_c1\":float,\"x0_c1\":float,\"x1_c0\":float,\"x0_c0\":float}}. "
    "p_conf = P(confounder present). p_y.x1_c1 = P(outcome present | treatment "
    "present, confounder present); x0_c1 = treatment absent, confounder present; "
    "x1_c0 = treatment present, confounder absent; x0_c0 = both absent. Convert "
    "percentages to [0,1] decimals. Use the population's stated causal direction.")

_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
def pj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None

def the_one_estimate(ext, consistent):
    """Build a graph from gpt-5.1's extracted numbers; engine marginalizes exactly.
    Returns (estimate, ok). Protocol failure (bad extraction) -> ok=False (AM-007)."""
    try:
        pc = float(ext["p_conf"]); py = ext["p_y"]
        rows = {k: float(py[k]) for k in ("x1_c1", "x0_c1", "x1_c0", "x0_c0")}
        if not (0 <= pc <= 1 and all(0 <= x <= 1 for x in rows.values())):
            return None, False
        g = CausalGraph()
        for n in ("U", "X", "Y"):
            g.add_variable(Variable(n))
        g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
        g.set_cpt("U", {(): {1: round(pc, 6), 0: round(1 - pc, 6)}})
        # X-mechanism unknown to extractor (distractor, irrelevant to do(X=1));
        # any valid CPT works since we intervene on X. Use neutral 0.5.
        g.set_cpt("X", {(1,): {1: 0.5, 0: 0.5}, (0,): {1: 0.5, 0: 0.5}})
        oY = list(g.parent_order("Y"))
        m = {(1, 1): rows["x1_c1"], (0, 1): rows["x0_c1"],
             (1, 0): rows["x1_c0"], (0, 0): rows["x0_c0"]}
        g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY):
                        {1: round(val, 6), 0: round(1 - val, 6)} for (u, x), val in m.items()})
        est = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
        return round(est, 6), True
    except Exception:
        return None, False

def main():
    jpath = HERE / "rows.jsonl"
    done = {json.loads(l)["i"] for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jf = jpath.open("a")
    rows = []
    for i in range(N_ITEMS):
        X, Y, U, consistent = WORDS[i % len(WORDS)]
        rng = np.random.default_rng(SEED + 1 + i)
        g, pu, px1, px0, v, do, obs = make_scm(rng)
        if i in done:
            continue
        narr = narrate(X, Y, U, consistent, pu, px1, px0, v)
        sys = "You are a careful causal-inference expert."
        # RAW arm: gpt-5.1 end-to-end
        try:
            raw_txt = _openai("gpt-5.1", sys, narr + "\n\n" + SOLVE.format(X=X, Y=Y))
            mm = list(_ANS.finditer(raw_txt))
            raw = float(mm[-1].group(1)) if mm else None
        except Exception:
            raw = None
        # The One arm: gpt-5.1 extracts numbers -> engine marginalizes
        try:
            ext = pj(_openai("gpt-5.1", sys, narr + "\n\n" + EXTRACT))
        except Exception:
            ext = None
        one, one_ok = the_one_estimate(ext, consistent) if ext else (None, False)
        row = {"i": i, "consistent": consistent, "X": X, "Y": Y, "U": U,
               "truth_do": do, "obs": obs, "gap": round(abs(do - obs), 4),
               "raw": raw, "the_one": one, "the_one_ok": one_ok,
               "narrative": narr}
        rows.append(row); jf.write(json.dumps(row) + "\n"); jf.flush()
        re_ = (abs(raw - do) if raw is not None else None)
        oe_ = (abs(one - do) if one is not None else None)
        print(f"[{i:02d}] {'CN' if consistent else 'CT'} truth={do:.3f} obs={obs:.3f} "
              f"| raw={raw} err={re_ if re_ is None else round(re_,3)} "
              f"| one={one} err={oe_ if oe_ is None else round(oe_,3)} ok={one_ok}", flush=True)

    # full corpus for summary
    all_rows = [json.loads(l) for l in jpath.read_text().splitlines()]
    def acc(rs, key, tol):
        ok = [r for r in rs if r.get(key) is not None]
        return (round(sum(1 for r in ok if abs(r[key] - r["truth_do"]) <= tol) / len(ok), 3)
                if ok else None, len(ok))
    def mae(rs, key):
        ok = [r for r in rs if r.get(key) is not None]
        return round(float(np.mean([abs(r[key] - r["truth_do"]) for r in ok])), 4) if ok else None
    # naive-confounded baseline: how often raw equals the OBSERVATIONAL (wrong) answer
    raw_ok = [r for r in all_rows if r.get("raw") is not None]
    fell_for_trap = round(sum(1 for r in raw_ok if abs(r["raw"] - r["obs"]) < abs(r["raw"] - r["truth_do"]))
                          / len(raw_ok), 3) if raw_ok else None
    summary = {
        "n": len(all_rows),
        "raw_acc_strict": acc(all_rows, "raw", TOL), "raw_acc_loose": acc(all_rows, "raw", LOOSE),
        "raw_mae": mae(all_rows, "raw"),
        "the_one_acc_strict": acc(all_rows, "the_one", TOL),
        "the_one_acc_loose": acc(all_rows, "the_one", LOOSE),
        "the_one_mae": mae(all_rows, "the_one"),
        "the_one_extract_valid": round(sum(1 for r in all_rows if r.get("the_one_ok")) / len(all_rows), 3),
        "raw_closer_to_obs_than_truth(trap)": fell_for_trap,
        "mean_gap": round(float(np.mean([r["gap"] for r in all_rows])), 4),
    }
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": all_rows}, indent=2))
    print("\n===== WILDTEXT-NUMERIC SUMMARY =====")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
