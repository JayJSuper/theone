"""Semi-structured wild-text, k=2 (TWO independent confounders). Same isolation
as run.py (k=1): SAME front-end gpt-5.1 both arms, only marginalization differs.
Marginalization load rises 2 configs -> 4 configs. Tests whether raw in-head
arithmetic degrades with load (connecting the low-k toe to the combinatorial
cliff) while the engine stays exact.

Run: source ~/.theone_keys.env && python experiments/wildtext_numeric/run_k2.py
"""
from __future__ import annotations
import json, os, re, urllib.request
from pathlib import Path
import numpy as np
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
SEED = 20260620
N_ITEMS = 24
TOL, LOOSE = 0.005, 0.05
DELTA = 0.05

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

WORDS = [  # (X, Y, U1, U2, consistent?)
    ("daily walking", "low resting heart rate", "older age", "high body weight", True),
    ("high vitamin D", "more outdoor time", "low latitude", "warm season", False),
    ("online tutoring", "high exam score", "high family income", "small class size", True),
    ("frequent code review", "low bug count", "high module complexity", "junior team", False),
    ("fertilizer use", "high crop yield", "good soil quality", "high rainfall", True),
    ("umbrella carrying", "rainfall", "winter season", "coastal region", False),
    ("standing-desk use", "low back pain", "high baseline fitness", "young age", True),
    ("ice-cream consumption", "high temperature", "summer month", "holiday period", False),
]

def pct(p): return round(100 * p)

def build_graph(p1, p2, pyrows):
    """pyrows: dict keyed (u1,u2)->P(Y=1|X=1,u1,u2) AND we need X-dependence too.
    Here pyrows maps (x,u1,u2)->P(Y=1). Returns engine-ready graph."""
    g = CausalGraph()
    for n in ("U1", "U2", "X", "Y"):
        g.add_variable(Variable(n))
    for u in ("U1", "U2"):
        g.add_edge(u, "X"); g.add_edge(u, "Y")
    g.add_edge("X", "Y")
    g.set_cpt("U1", {(): {1: round(p1, 6), 0: round(1 - p1, 6)}})
    g.set_cpt("U2", {(): {1: round(p2, 6), 0: round(1 - p2, 6)}})
    return g

def make_scm(rng):
    for _ in range(400):
        p1 = round(float(rng.uniform(.3, .7)), 2)
        p2 = round(float(rng.uniform(.3, .7)), 2)
        g = build_graph(p1, p2, None)
        # X | U1,U2 (distractor) — strong dependence so confounding is real
        oX = list(g.parent_order("X"))
        xv = {(1, 1): round(float(rng.uniform(.70, .92)), 2),
              (1, 0): round(float(rng.uniform(.45, .70)), 2),
              (0, 1): round(float(rng.uniform(.30, .55)), 2),
              (0, 0): round(float(rng.uniform(.08, .30)), 2)}
        g.set_cpt("X", {tuple(a if p == "U1" else b for p in oX):
                        {1: v, 0: round(1 - v, 2)} for (a, b), v in xv.items()})
        # Y | X,U1,U2
        oY = list(g.parent_order("Y"))
        yv = {}
        for x in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    base = .15 + .35 * x + .18 * a + .14 * b
                    val = round(min(.93, max(.06, base + float(rng.uniform(-.08, .08)))), 2)
                    yv[(x, a, b)] = val
        g.set_cpt("Y", {tuple({"X": x, "U1": a, "U2": b}[p] for p in oY):
                        {1: v, 0: round(1 - v, 2)} for (x, a, b), v in yv.items()})
        eng = InterventionEngine(g)
        do = eng.query_intervention("Y", 1, {"X": 1}).value
        obs = eng.query_observation("Y", 1, {"X": 1}).value
        if abs(do - obs) >= DELTA:
            return g, p1, p2, xv, yv, round(do, 6), round(obs, 6)
    raise RuntimeError("no k2 SCM with gap")

def narrate(X, Y, U1, U2, consistent, p1, p2, yv):
    arrow = (f"In this population {X} causally drives {Y} (earlier in time, changes "
             f"{Y} downstream).")
    cn = ("Consistent with common sense." if consistent else
          "NOTE: this REVERSES real-world common sense — honor the stated direction anyway.")
    def grp(a, b, label):
        return (f"Among those {label}: {pct(yv[(1,a,b)])}% of {X}-exposed showed {Y}, "
                f"versus {pct(yv[(0,a,b)])}% of unexposed.")
    return (
        f"A cohort registry studied whether {X} affects {Y}, recording two common "
        f"causes of both: {U1} and {U2}. {arrow} {cn} "
        f"{U1} was present in {pct(p1)}% of the cohort and {U2} in {pct(p2)}% "
        f"(the two were recorded independently). "
        f"{grp(1,1,f'with BOTH {U1} and {U2}')} "
        f"{grp(1,0,f'with {U1} but not {U2}')} "
        f"{grp(0,1,f'with {U2} but not {U1}')} "
        f"{grp(0,0,f'with NEITHER {U1} nor {U2}')} "
        f"(The treatment's own prevalence varied across these strata but is not "
        f"the question here.)")

SOLVE = (
    "Estimate P({Y}=present | do({X}=present)) — probability of the outcome if {X} "
    "were SET to present for everyone by intervention, in THIS population's stated "
    "direction. Adjust for BOTH confounders correctly (intervention, not "
    "observation). Reason step by step, then end with exactly: "
    "ANSWER: <number 0-1, 4 decimals>.")

EXTRACT = (
    "Extract as JSON ONLY: {\"p_c1\":float,\"p_c2\":float,\"p_y\":{"
    "\"x1_11\":f,\"x0_11\":f,\"x1_10\":f,\"x0_10\":f,\"x1_01\":f,\"x0_01\":f,"
    "\"x1_00\":f,\"x0_00\":f}}. p_c1=P(first confounder present), p_c2=P(second). "
    "p_y.xA_BC = P(outcome present | treatment=A, conf1=B, conf2=C), A/B/C in {0,1}. "
    "Percentages -> [0,1] decimals. Use the stated causal direction.")

_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
def pj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try: return json.loads(m.group(0)) if m else None
    except Exception: return None

def the_one(ext):
    try:
        p1 = float(ext["p_c1"]); p2 = float(ext["p_c2"]); py = ext["p_y"]
        keys = ["x1_11","x0_11","x1_10","x0_10","x1_01","x0_01","x1_00","x0_00"]
        r = {k: float(py[k]) for k in keys}
        if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and all(0 <= v <= 1 for v in r.values())):
            return None, False
        g = build_graph(p1, p2, None)
        oX = list(g.parent_order("X"))
        g.set_cpt("X", {tuple(a if p == "U1" else b for p in oX): {1: 0.5, 0: 0.5}
                        for a in (0, 1) for b in (0, 1)})
        oY = list(g.parent_order("Y"))
        def k(x, a, b): return f"x{x}_{a}{b}"
        g.set_cpt("Y", {tuple({"X": x, "U1": a, "U2": b}[p] for p in oY):
                        {1: round(r[k(x,a,b)], 6), 0: round(1 - r[k(x,a,b)], 6)}
                        for x in (0,1) for a in (0,1) for b in (0,1)})
        est = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
        return round(est, 6), True
    except Exception:
        return None, False

def main():
    jpath = HERE / "rows_k2.jsonl"
    done = {json.loads(l)["i"] for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jf = jpath.open("a")
    for i in range(N_ITEMS):
        X, Y, U1, U2, consistent = WORDS[i % len(WORDS)]
        g, p1, p2, xv, yv, do, obs = make_scm(np.random.default_rng(SEED + 1 + i))
        if i in done:
            continue
        narr = narrate(X, Y, U1, U2, consistent, p1, p2, yv)
        sys = "You are a careful causal-inference expert."
        try:
            txt = _openai("gpt-5.1", sys, narr + "\n\n" + SOLVE.format(X=X, Y=Y))
            mm = list(_ANS.finditer(txt)); raw = float(mm[-1].group(1)) if mm else None
        except Exception:
            raw = None
        try:
            ext = pj(_openai("gpt-5.1", sys, narr + "\n\n" + EXTRACT))
        except Exception:
            ext = None
        one, ok = the_one(ext) if ext else (None, False)
        row = {"i": i, "consistent": consistent, "truth_do": do, "obs": obs,
               "gap": round(abs(do-obs), 4), "raw": raw, "the_one": one,
               "the_one_ok": ok, "narrative": narr}
        jf.write(json.dumps(row) + "\n"); jf.flush()
        re_ = None if raw is None else round(abs(raw-do), 3)
        oe_ = None if one is None else round(abs(one-do), 3)
        print(f"[{i:02d}] {'CN' if consistent else 'CT'} truth={do:.3f} obs={obs:.3f} "
              f"| raw={raw} err={re_} | one={one} err={oe_} ok={ok}", flush=True)

    rows = [json.loads(l) for l in jpath.read_text().splitlines()]
    def acc(key, tol):
        ok = [r for r in rows if r.get(key) is not None]
        return (round(sum(1 for r in ok if abs(r[key]-r["truth_do"]) <= tol)/len(ok), 3), len(ok)) if ok else (None,0)
    def mae(key):
        ok = [r for r in rows if r.get(key) is not None]
        return round(float(np.mean([abs(r[key]-r["truth_do"]) for r in ok])), 4) if ok else None
    summary = {"k": 2, "n": len(rows),
               "raw_acc_strict": acc("raw", TOL), "raw_acc_loose": acc("raw", LOOSE),
               "raw_mae": mae("raw"), "raw_max": round(max((abs(r["raw"]-r["truth_do"]) for r in rows if r.get("raw") is not None), default=0),4),
               "the_one_acc_strict": acc("the_one", TOL), "the_one_mae": mae("the_one"),
               "the_one_extract_valid": round(sum(1 for r in rows if r.get("the_one_ok"))/len(rows), 3),
               "mean_gap": round(float(np.mean([r["gap"] for r in rows])), 4)}
    (HERE / "results_k2.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n===== WILDTEXT-NUMERIC k=2 SUMMARY =====")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
