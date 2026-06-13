"""Wild-text axis PILOT (frozen prereg cd49adf): can S1-rewritten narratives
carry causal structure losslessly? Dual-LLM extraction (deepseek + gpt-5.1)
+ structure-level back-translation (CC amendment to Q-C17-2).
Run: source ~/.theone_keys.env && python experiments/wildtext_axis/pilot.py
"""
from __future__ import annotations
import json
import os
import re
import time
import urllib.request
from pathlib import Path
import numpy as np
from theone.llm import DeepSeekClient

HERE = Path(__file__).parent
SEED = 20260616

# frozen word bank: (treatment, outcome, confounder, distractor)
BANK = [
    ("daily walking", "resting heart rate", "age", "neighborhood tree cover"),
    ("online tutoring", "exam scores", "parental income", "textbook brand"),
    ("fertilizer use", "crop yield", "soil quality", "tractor color"),
    ("standing desks", "back pain", "baseline fitness", "office wall color"),
    ("meditation app use", "sleep quality", "work stress", "phone brand"),
    ("bike commuting", "BMI", "city density", "helmet style"),
    ("library visits", "reading ability", "household education", "library font"),
    ("vitamin D intake", "bone density", "outdoor time", "pill color"),
    ("code review", "bug rate", "team experience", "IDE theme"),
    ("irrigation", "plant growth", "rainfall", "fence height"),
    ("flu vaccination", "sick days", "health consciousness", "clinic distance"),
    ("loyalty programs", "repeat purchases", "income level", "logo design"),
]

REWRITE_L1 = (
    "Rewrite the following causal facts as ONE natural news-style paragraph "
    "(3-5 sentences). You MUST explicitly state that {conf} influences both "
    "{treat} and {out}, and that {treat} influences {out}. Also mention {dist} "
    "as a recorded but causally irrelevant detail. Do NOT add any other causal "
    "claims. No numbers needed.")
REWRITE_L2 = (
    "Rewrite the following causal facts as ONE natural news-style paragraph "
    "(3-5 sentences). Mention that {conf} was recorded and describe its "
    "associations OBSERVATIONALLY (e.g. 'people with higher {conf} also tended "
    "to...') WITHOUT labelling it as a cause or confounder. State that {treat} "
    "and {out} are associated. Mention {dist} as an irrelevant recorded detail. "
    "Do NOT add any other causal claims. No numbers needed.")
EXTRACT = (
    "Read the study description. Identify the causal roles. Reply with ONLY a "
    "JSON object, no prose: {\"treatment\": str, \"outcome\": str, "
    "\"confounders\": [str], \"adjustment_set\": [str]} . The "
    "adjustment_set is what you would control for to estimate the causal "
    "effect of treatment on outcome - confounders belong in BOTH lists. "
    "Use short noun "
    "phrases copied from the text. If a variable merely appears but plays no "
    "causal role, exclude it.")


def ask_gpt51(system, user):
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps({"model": "gpt-5.1",
                         "messages": [{"role": "system", "content": system},
                                      {"role": "user", "content": user}],
                         "max_completion_tokens": 1024}).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def fuzzy(a, b):
    """Word-set Jaccard >= 0.5 OR containment (noun-phrase paraphrase tolerant)."""
    A, B = set(norm(a).split()), set(norm(b).split())
    if not A or not B:
        return False
    if norm(a) in norm(b) or norm(b) in norm(a):
        return True
    return len(A & B) / len(A | B) >= 0.5


def role_match(ext, treat, out, conf):
    """Structure-level back-translation: extracted roles == true roles."""
    if ext is None:
        return False, "unparseable"
    def hit(field, target):
        return fuzzy(ext.get(field, ""), target)
    ok_t = hit("treatment", treat)
    ok_o = hit("outcome", out)
    adj = ext.get("adjustment_set", []) or ext.get("confounders", [])
    ok_a = (len(adj) >= 1 and any(fuzzy(a, conf) for a in adj)
            and all(fuzzy(a, conf) for a in adj))  # exactly the confounder
    reasons = []
    if not ok_t: reasons.append("treatment")
    if not ok_o: reasons.append("outcome")
    if not ok_a: reasons.append(f"adjustment_set={ext.get('adjustment_set')}")
    return (ok_t and ok_o and ok_a), (",".join(reasons) or "ok")


def main():
    s1 = DeepSeekClient()
    rng = np.random.default_rng(SEED)
    rows = []
    for li, level in enumerate(("L1", "L2")):
        tmpl = REWRITE_L1 if level == "L1" else REWRITE_L2
        for bi, (treat, out, conf, dist) in enumerate(BANK):
            facts = (f"Causal facts: {conf} -> {treat}; {conf} -> {out}; "
                     f"{treat} -> {out}. Irrelevant recorded variable: {dist}.")
            try:
                narrative = s1.chat(
                    [{"role": "system", "content": tmpl.format(
                        treat=treat, out=out, conf=conf, dist=dist)},
                     {"role": "user", "content": facts}],
                    max_tokens=400, temperature=0.7)["content"].strip()
            except Exception as e:
                rows.append({"level": level, "bank": bi, "accept": False,
                             "why": f"rewrite failed: {e}"[:100]})
                continue
            # dual extraction
            try:
                e1 = parse_json(s1.chat(
                    [{"role": "system", "content": EXTRACT},
                     {"role": "user", "content": narrative}],
                    max_tokens=2048, temperature=0.0)["content"])
            except Exception:
                e1 = None
            try:
                e2 = parse_json(ask_gpt51(EXTRACT, narrative))
            except Exception:
                e2 = None
            ok1, why1 = role_match(e1, treat, out, conf)
            ok2, why2 = role_match(e2, treat, out, conf)
            def adjs(e):
                return e.get("adjustment_set", []) or e.get("confounders", [])
            agree = (e1 is not None and e2 is not None and
                     fuzzy(e1.get("treatment", ""), e2.get("treatment", "")) and
                     len(adjs(e1)) == len(adjs(e2)) and
                     all(any(fuzzy(a, b) for b in adjs(e2)) for a in adjs(e1)))
            accept = ok1 and ok2 and agree
            rows.append({"level": level, "bank": bi, "accept": accept,
                         "agree": agree, "ds_ok": ok1, "gpt_ok": ok2,
                         "ds_why": why1, "gpt_why": why2,
                         "narrative": narrative[:400]})
            print(f"[{level}-{bi:02d}] accept={accept}  ds={why1}  gpt={why2}  "
                  f"agree={agree}", flush=True)

    res = {"seed": SEED}
    for level in ("L1", "L2"):
        lr = [r for r in rows if r["level"] == level]
        res[level] = {"n": len(lr),
                      "acceptance": round(sum(r["accept"] for r in lr) / len(lr), 3)}
    res["threshold"] = 0.85
    res["verdict"] = {lv: ("automatable" if res[lv]["acceptance"] >= 0.85
                           else "below_threshold") for lv in ("L1", "L2")}
    (HERE / "results_v2.json").write_text(json.dumps(
        {"summary": res, "rows": rows}, indent=2, ensure_ascii=False))
    print("\n===== WILDTEXT PILOT SUMMARY =====")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
