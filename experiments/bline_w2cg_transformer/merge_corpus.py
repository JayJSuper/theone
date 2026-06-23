"""Robustly parse + merge the raw LLM-generated corpora into one clean labeled corpus.

DeepSeek echoed field labels inline; Gemini gave clean 4-field lines. We parse by SET MEMBERSHIP
(find the valid CAUSE / EFFECT / DIR token in each pipe-split line, wherever it sits), validate
against the broadened schema, dedup, and write a clean corpus the transformer trainer consumes.
"""
from __future__ import annotations
import re
from pathlib import Path

CAUSES = {"smoking", "exercise", "drug", "alcohol", "sleep", "diet", "stress", "vaccine", "none"}
EFFECTS = {"cancer", "mortality", "recovery", "heart_disease", "diabetes", "depression", "infection", "none"}
DIRS = {"+", "-", "0", "none"}
HERE = Path(__file__).parent
RAW = ["/tmp/w2cg_big_deepseek.txt", "/tmp/w2cg_big_gemini.txt",
              "/tmp/w2cg_gen2.txt", "/tmp/w2cg_gen3.txt",
       "/tmp/w2cg_genx1.txt", "/tmp/w2cg_genx2.txt", "/tmp/w2cg_genx3.txt", "/tmp/w2cg_genx4.txt", "/tmp/w2cg_genx5.txt",
       "/tmp/w2cg_genz0.txt", "/tmp/w2cg_genz1.txt", "/tmp/w2cg_genz2.txt", "/tmp/w2cg_genz3.txt", "/tmp/w2cg_genz4.txt", "/tmp/w2cg_genz5.txt", "/tmp/w2cg_genz6.txt", "/tmp/w2cg_genz7.txt", "/tmp/w2cg_genz8.txt", "/tmp/w2cg_genz9.txt",
       "/tmp/w2cg_genw0.txt", "/tmp/w2cg_genw1.txt", "/tmp/w2cg_genw2.txt", "/tmp/w2cg_genw3.txt", "/tmp/w2cg_genw4.txt", "/tmp/w2cg_genw5.txt", "/tmp/w2cg_genw6.txt", "/tmp/w2cg_genw7.txt", "/tmp/w2cg_genw8.txt", "/tmp/w2cg_genw9.txt", "/tmp/w2cg_genw10.txt", "/tmp/w2cg_genw11.txt"]
SEED = HERE.parent / "bline_w2cg_learned" / "corpus.txt"      # original 123 (narrow schema, still valid)


# unambiguous surface cues — used ONLY to repair a dropped (none) label, never to override one
CAUSE_CUE = {"smoking": ["lighting up", "light up", "puffing", "cigarette", " cig", "tobacco", "smoke", "pack a day"],
             "alcohol": ["booz", "drink like a fish", " beer", " wine", "a few drinks", "heavy drinking"],
             "exercise": ["gym", "working out", "work out", "break a sweat", "breaking a sweat", "jogging", "cardio"],
             "vaccine": ["vaccin", "jab", "shot ", "immuniz"], "sleep": ["sleep", "z's", "z’s", "rest", "shut-eye"],
             "stress": ["stress", "anxiet", "burnout"], "diet": ["diet", "junk food", "veggies", "eating "]}
EFFECT_CUE = {"cancer": ["cancer", "tumor", "tumour", "the big c"], "heart_disease": ["ticker", "heart attack", "heart disease", "cardiac"],
              "depression": ["depress", "the blues", "feeling down"], "mortality": ["six feet under", "die", "death", "live longer", "kick the bucket"],
              "diabetes": ["diabet", "blood sugar"], "infection": ["infect", "the flu", "a cold", "virus"], "recovery": ["recover", "getting better", "heal"]}


def _repair(label, sent, cue):
    if label and label != "none":
        return label
    s = sent.lower()
    for canon, cues in cue.items():
        if any(c in s for c in cues):
            return canon
    return label or "none"


def parse_line(line: str):
    segs = [s.strip() for s in line.split("|")]
    cause = next((s for s in segs if s.lower() in CAUSES and s.lower() != "none"), None)
    effect = next((s for s in segs if s.lower() in EFFECTS and s.lower() != "none"), None)
    direction = next((s for s in segs if s in DIRS), None)
    # sentence = the 'sentence:'-prefixed segment, else the longest free-text segment
    sent = None
    for s in segs:
        if s.lower().startswith("sentence:"):
            sent = s.split(":", 1)[1].strip(); break
    if sent is None:
        cands = [s for s in segs if len(s.split()) >= 3 and s.lower() not in CAUSES | EFFECTS]
        sent = max(cands, key=len) if cands else None
    if not sent or direction is None or (cause is None and effect is None):
        return None
    c = _repair((cause or "none").lower(), sent, CAUSE_CUE)
    e = _repair((effect or "none").lower(), sent, EFFECT_CUE)
    return c, e, direction, sent


def main():
    rows = {}
    for path in RAW:
        p = Path(path)
        if not p.exists():
            continue
        for ln in p.read_text().splitlines():
            if "|" not in ln:
                continue
            r = parse_line(ln)
            if r:
                rows[r[3].lower()] = r            # dedup by sentence text
    # seed corpus (clean 4-field, narrow schema)
    for ln in SEED.read_text().splitlines():
        parts = [x.strip() for x in ln.split("|", 3)]
        if len(parts) == 4 and parts[2] in DIRS:
            rows[parts[3].lower()] = (parts[0], parts[1], parts[2], parts[3])
    out = HERE / "corpus.txt"
    lines = [f"{c} | {e} | {d} | {s}" for (c, e, d, s) in rows.values()]
    out.write_text("\n".join(lines) + "\n")
    # quick label distribution
    from collections import Counter
    cc = Counter(r[0] for r in rows.values()); ec = Counter(r[1] for r in rows.values())
    print(f"merged {len(lines)} unique labeled sentences -> {out}")
    print(f"  causes: {dict(cc)}")
    print(f"  effects: {dict(ec)}")


if __name__ == "__main__":
    main()
