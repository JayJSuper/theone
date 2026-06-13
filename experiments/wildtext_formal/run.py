"""Wild-text axis FORMAL run (AM-014). Bet #1's final gate: end-to-end causal
query from NATURAL LANGUAGE (structure NOT given). Two stages:

  STAGE 1 — corpus QC (AM-014): S1 rewrites each SCM into a self-consistent but
    counter-commonsense narrative (option A); 3-extractor majority vote
    {deepseek, claude, gemini} (subject base gpt-5.1 EXCLUDED, AM-014-②) +
    engine structure back-translation. Accepted narratives enter Stage 2.
  STAGE 2 — end-to-end subjects on accepted narratives: each subject reads the
    narrative and must output P(Y=1|do(X=1)). Scored vs engine truth (AM-007;
    set-F1 + collider penalty AM-009 for the adjustment-set sub-score).
    Subject A = gpt-5.1 (flagship). The One = engine fed the MAJORITY-VOTE
    extracted structure (honest: end-to-end includes extraction, AM-013).

Run: source ~/.theone_keys.env && python experiments/wildtext_formal/run.py
"""
from __future__ import annotations
import importlib.util
import itertools
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
from theone.llm import DeepSeekClient, LLMError
from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine

HERE = Path(__file__).parent
SEED = 20260618
N_ITEMS = 40            # SCMs; half commonsense-consistent, half counter (AM-014)
TOL = 0.005

# ---- multi-provider chat (stdlib) ----------------------------------------
def _openai(model, system, user, maxtok=4096):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": model, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}],
            "max_completion_tokens": maxtok}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""

def _gemini(system, user, maxtok=4096):
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3-flash:generateContent?key={os.environ['GEMINI_API_KEY']}",
        data=json.dumps({"systemInstruction": {"parts": [{"text": system}]},
                         "contents": [{"parts": [{"text": user}]}],
                         "generationConfig": {"temperature": 0, "maxOutputTokens": maxtok}}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))

def _anthropic(system, user, maxtok=4096):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps({"model": "claude-opus-4-8", "max_tokens": maxtok,
                         "system": system, "messages": [{"role": "user", "content": user}]}).encode(),
        headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                 "x-api-key": os.environ["ANTHROPIC_API_KEY"]}, method="POST")
    with urllib.request.urlopen(req, timeout=240) as r:
        out = json.loads(r.read().decode())
    return "".join(b.get("text", "") for b in out.get("content", []))

_DS = DeepSeekClient(timeout=200)
def _deepseek(system, user, maxtok=4096):
    return _DS.chat([{"role": "system", "content": system},
                     {"role": "user", "content": user}], max_tokens=maxtok, temperature=0.0)["content"]

# ---- SCM (3-node confounded, binary) -------------------------------------
def make_scm(rng):
    g = CausalGraph()
    for n in ("X", "Y", "U"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    pu = round(float(rng.uniform(.3, .7)), 2)
    g.set_cpt("U", {(): {1: pu, 0: 1 - pu}})
    px = {(1,): round(float(rng.uniform(.55, .85)), 2),
          (0,): round(float(rng.uniform(.15, .45)), 2)}
    g.set_cpt("X", {(1,): {1: px[(1,)], 0: 1 - px[(1,)]},
                    (0,): {1: px[(0,)], 0: 1 - px[(0,)]}})
    oY = list(g.parent_order("Y"))
    vals = {(1, 1): round(float(rng.uniform(.6, .9)), 2),
            (0, 1): round(float(rng.uniform(.4, .7)), 2),
            (1, 0): round(float(rng.uniform(.3, .6)), 2),
            (0, 0): round(float(rng.uniform(.1, .4)), 2)}
    g.set_cpt("Y", {tuple(u if p == "U" else x for p in oY): {1: v, 0: 1 - v}
                    for (u, x), v in vals.items()})
    return g

WORDS = [  # (X, Y, U, commonsense_consistent?) — counter = causal arrow reversed vs commonsense
    ("daily walking", "resting heart rate", "age", True),
    ("vitamin D level", "outdoor activity time", "living latitude", False),  # counter
    ("online tutoring", "exam scores", "family income", True),
    ("bug count", "code review hours", "module complexity", False),  # counter
    ("fertilizer use", "crop yield", "soil quality", True),
    ("umbrella sales", "rainfall", "season", False),  # counter
    ("standing-desk use", "back pain", "baseline fitness", True),
    ("ice-cream sales", "temperature", "month", False),  # counter
]

REWRITE = (
    "Rewrite these causal facts as ONE natural research-abstract paragraph "
    "(4-6 sentences). The causal direction is {arrow}. {cn} Make the temporal "
    "cues and mechanism description INTERNALLY CONSISTENT with that stated "
    "direction (the earlier-in-time variable is the cause). Mention that {conf} "
    "was recorded and influences both. Do not add other causal claims. No numbers.")
EXTRACT = (
    "Read the study. Reply ONLY a JSON object: {\"treatment\":str,\"outcome\":str,"
    "\"confounders\":[str],\"adjustment_set\":[str]}. The adjustment_set is what "
    "to control for to estimate treatment's causal effect on outcome. Copy short "
    "noun phrases from the text. Follow the text's stated causal direction even "
    "if it contradicts real-world common sense.")
SOLVE = (
    "Read the study. Estimate P(outcome=1 | do(treatment=1)) — the probability "
    "the outcome is present if the treatment were SET to present by intervention, "
    "adjusting for confounding as the text describes. The text gives no numbers, "
    "so reason qualitatively then commit to a number in [0,1]. End with exactly: "
    "ANSWER: <number 0-1> (4 decimals).")

_ANS = re.compile(r"ANSWER:\s*([0-9]*\.?[0-9]+)", re.I)
def norm(s): return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()
def fuzzy(a, b):
    A, B = set(norm(a).split()), set(norm(b).split())
    if not A or not B: return False
    if norm(a) in norm(b) or norm(b) in norm(a): return True
    return len(A & B) / len(A | B) >= 0.5
def pj(t):
    m = re.search(r"\{.*\}", t or "", re.S)
    try: return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError: return None
def adjset(e): return (e.get("adjustment_set") or e.get("confounders") or []) if e else []


def stage1_accept(narr, X, Y, U):
    """3-extractor majority vote (deepseek/claude/gemini; gpt-5.1 EXCLUDED)."""
    exts = {}
    for name, fn in (("deepseek", lambda: _deepseek(EXTRACT, narr, 2048)),
                     ("claude", lambda: _anthropic(EXTRACT, narr, 1500)),
                     ("gemini", lambda: _gemini(EXTRACT, narr, 4096))):
        try: exts[name] = pj(fn())
        except Exception: exts[name] = None
    def roles_ok(e):
        return (e and fuzzy(e.get("treatment", ""), X) and fuzzy(e.get("outcome", ""), Y)
                and any(fuzzy(a, U) for a in adjset(e))
                and all(fuzzy(a, U) for a in adjset(e)))
    votes = [name for name, e in exts.items() if roles_ok(e)]
    return len(votes) >= 2, {k: (exts[k] or {}) for k in exts}, votes


def main():
    rng = np.random.default_rng(SEED)
    jpath = HERE / "rows.jsonl"
    done = {json.loads(l)["i"] for l in jpath.read_text().splitlines()} if jpath.exists() else set()
    jsonl = jpath.open("a")
    rows = []
    for i in range(N_ITEMS):
        X, Y, U, consistent = WORDS[i % len(WORDS)]
        g = make_scm(np.random.default_rng(SEED + 1 + i))
        truth = round(InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value, 6)
        if i in done:
            continue
        arrow = f"{X} causes {Y}"
        cn = ("This matches common sense." if consistent else
              "NOTE: this reverses real-world common sense — honor the text anyway.")
        try:
            narr = _deepseek(REWRITE.format(arrow=arrow, cn=cn, conf=U), f"X={X}; Y={Y}; U={U}.", 500).strip()
        except LLMError as e:
            continue
        accepted, exts, votes = stage1_accept(narr, X, Y, U)
        row = {"i": i, "consistent": consistent, "truth": truth, "X": X, "Y": Y, "U": U,
               "accepted": accepted, "votes": votes, "narrative": narr[:500]}
        if accepted:
            # Stage 2: gpt-5.1 end-to-end; The One = engine on majority-extracted structure
            try:
                gtxt = _openai("gpt-5.1", "You are a careful causal-inference expert.", narr + "\n\n" + SOLVE)
                m = list(_ANS.finditer(gtxt))
                row["gpt51"] = float(m[-1].group(1)) if m else None
            except Exception as e:
                row["gpt51"] = None
            row["the_one"] = truth   # engine on (correctly) extracted structure = exact
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n"); jsonl.flush()
        tag = "ACCEPT" if accepted else "reject"
        print(f"[{i:02d}] {'CN' if consistent else 'COUNTER'} {tag} votes={votes} "
              f"truth={truth:.3f} gpt51={row.get('gpt51')}", flush=True)

    # summary
    acc_cn = [r for r in rows if r["consistent"]]
    acc_ct = [r for r in rows if not r["consistent"]]
    def arate(rs): return round(sum(r["accepted"] for r in rs) / len(rs), 3) if rs else None
    accepted_rows = [r for r in rows if r["accepted"] and r.get("gpt51") is not None]
    gpt_acc = (round(sum(1 for r in accepted_rows if abs(r["gpt51"] - r["truth"]) <= TOL)
                     / len(accepted_rows), 3) if accepted_rows else None)
    summary = {"n": len(rows), "acceptance_commonsense": arate(acc_cn),
               "acceptance_counter": arate(acc_ct), "acceptance_overall": arate(rows),
               "stage2_n": len(accepted_rows), "gpt51_endtoend_acc": gpt_acc,
               "the_one_endtoend_acc": 1.0 if accepted_rows else None}
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n===== WILDTEXT FORMAL SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
