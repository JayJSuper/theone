"""Propose-and-verify — the safe way to expand VERIFIED coverage on real language.

The peers' verdict (NOTE-117): a learned model can never carry the never-false-verify red-line;
VERIFIED must be anchored by an independent exact check. So: the learned proposer FINDS a
candidate (cause, effect, direction) — high recall, handles idioms — and an independent GROUNDING
verifier DISPOSES: VERIFY only if the proposed entities are actually present in the sentence (via
synonym match) AND a direction cue supports the claimed sign. Anything ungrounded -> abstain.

Coverage grows because the proposer resolves WHICH canonical entities (where the rule parser's
first-match heuristic guesses wrong), while grounding keeps the red-line: we never VERIFY entities
the text doesn't actually mention. We compare rule-alone vs propose-and-verify on the 4470 corpus.

Run:  .venv/bin/python experiments/bline_w2cg_propose_verify/run.py
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

CORPUS = Path(__file__).parent.parent / "bline_w2cg_transformer" / "corpus.txt"
CAUSES = ["none", "smoking", "exercise", "drug", "alcohol", "sleep", "diet", "stress", "vaccine"]
EFFECTS = ["none", "cancer", "mortality", "recovery", "heart_disease", "diabetes", "depression", "infection"]
DIRS = ["+", "-", "0", "none"]
SYN = {"smoking": ["smoking", "lighting up", "light up", "puffing", "cigarette", "cig", "tobacco", "smoke", "pack a day"],
       "alcohol": ["alcohol", "booz", "drink", "beer", "wine", "liquor"],
       "exercise": ["exercise", "gym", "working out", "work out", "sweat", "jogging", "cardio", "physical activity"],
       "drug": ["drug", "pill", "medication", "medicine"], "vaccine": ["vaccine", "vaccin", "jab", "flu shot", "immuniz"],
       "sleep": ["sleep", "z's", "rest", "shut-eye", "nap"], "stress": ["stress", "anxiet", "burnout", "tension"],
       "diet": ["diet", "junk food", "veggies", "eating", "fruit", "sugar"],
       "cancer": ["cancer", "tumor", "tumour", "the big c", "carcinoma"],
       "heart_disease": ["heart_disease", "ticker", "heart attack", "heart disease", "cardiac", "heart"],
       "depression": ["depression", "depress", "the blues", "feeling down", "mood"],
       "mortality": ["mortality", "six feet under", "die", "death", "live longer", "kick the bucket", "kicking the bucket", "grave"],
       "diabetes": ["diabetes", "diabet", "blood sugar"], "infection": ["infection", "infect", "the flu", "a cold", "cold", "virus", "flu"],
       "recovery": ["recovery", "recover", "getting better", "heal", "bounce back"]}
INC = ["increase", "increases", "raise", "raises", "boost", "boosts", "cause", "causes", "bump", "bumps",
       "skyrocket", "skyrockets", "improve", "improves", "speed", "speeds", "worsen", "raises", "more likely",
       "leads to", "lead to", "trigger", "triggers", "up your", "ups your", "wreck", "wrecks", "ruin"]
DEC = ["decrease", "decreases", "reduce", "reduces", "lower", "lowers", "cut", "cuts", "prevent", "prevents",
       "protect", "protects", "slash", "slashes", "keep", "keeps", "ward off", "fight", "fights", "guard", "less likely"]
NULL = ["no effect", "didn't do squat", "zero effect", "didn't change", "no impact", "doesn't change",
        "does not affect", "not affect", "unrelated", "no link", "nothing to do"]


def tok(s): return re.findall(r"[a-z']+", s.lower())


def load():
    rows = []
    for ln in CORPUS.read_text().splitlines():
        p = [x.strip() for x in ln.split("|", 3)]
        if len(p) == 4 and p[0] in CAUSES and p[1] in EFFECTS and p[2] in DIRS:
            rows.append((p[3], p[0], p[1], p[2]))
    return rows


def present(canon, s):                                   # is any synonym of canon in the sentence?
    s = " " + s.lower() + " "
    return any(w in s for w in SYN.get(canon, [canon]))


def dir_cue(s):                                          # directional cue actually in the text
    s = " " + s.lower() + " "
    neg = bool(re.search(r"\b(no|not|never|without)\b|n['’]t", s)) and not any(n in s for n in NULL)
    if any(n in s for n in NULL): return 0
    sign = None
    if any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in INC): sign = 1
    if any(re.search(r"\b" + re.escape(w) + r"\b", s) for w in DEC): sign = -1
    if "live longer" in s: sign = -1
    if neg and sign is not None: sign = 0
    return sign


def main():
    print("=== Propose-and-verify · safe VERIFIED-coverage expansion (4470 corpus) ===\n")
    rows = load()
    idx = np.random.default_rng(0).permutation(len(rows))
    cut = int(0.85 * len(rows)); tr = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]
    # fast bag-of-words proposer (3 heads)
    vocab = {}
    for s, *_ in tr:
        for t in tok(s):
            vocab.setdefault(t, len(vocab))
    V = len(vocab)
    def bow(s):
        x = np.zeros(V)
        for t in tok(s):
            if t in vocab: x[vocab[t]] = 1.0
        return x
    def head(labels, space):
        K = len(space); W = np.zeros((K, V)); b = np.zeros(K)
        X = np.stack([bow(s) for s, *_ in tr]); Y = np.array([space.index(l) for l in labels])
        for _ in range(250):
            z = X @ W.T + b; z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
            p[np.arange(len(Y)), Y] -= 1
            W -= 0.5 * (p.T @ X) / len(Y) + 1e-3 * W; b -= 0.5 * p.mean(0)
        return W, b
    Wc, bc = head([c for _, c, _, _ in tr], CAUSES); We, be = head([e for _, _, e, _ in tr], EFFECTS)
    Wd, bd = head([d for _, _, _, d in tr], DIRS)
    def propose(s):
        x = bow(s)
        return (CAUSES[int((Wc@x+bc).argmax())], EFFECTS[int((We@x+be).argmax())], DIRS[int((Wd@x+bd).argmax())])

    # structure from train (dominant dir per real edge)
    from collections import defaultdict, Counter
    edge = defaultdict(Counter)
    for _, c, e, d in tr:
        if c != "none" and e != "none" and d != "none": edge[(c, e)][d] += 1
    struct = {k: v.most_common(1)[0][0] for k, v in edge.items()}
    def gold(c, e, d):
        if c == "none" or e == "none" or d == "none" or (c, e) not in struct: return "UNVERIFIABLE"
        return "VERIFIED" if d == struct[(c, e)] else "CONTRADICTED"

    # rule-alone verifier (reuse product ClaimVerifier)
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    from theone.language import ClaimVerifier
    rstruct = {k: {"direction": {"+": 1, "-": -1, "0": 0}[d], "magnitude": None} for k, d in struct.items()}
    cv = ClaimVerifier(rstruct, SYN)

    # propose-and-verify: trust the proposal ONLY if grounded (entities present + dir cue supports)
    def pv_verdict(s):
        # entities FROM the proposer (resolves idioms), direction FROM the text (independent check)
        c, e, d = propose(s)
        if c == "none" or e == "none" or (c, e) not in struct: return "UNVERIFIABLE"
        if not (present(c, s) and present(e, s)): return "UNVERIFIABLE"   # red-line: entities must be in text
        cue = dir_cue(s)
        if cue is None: return "UNVERIFIABLE"                              # no grounded direction -> abstain
        sd = {"+": 1, "-": -1, "0": 0}[struct[(c, e)]]                     # struct dir is a string; cue is int
        return "VERIFIED" if cue == sd else "CONTRADICTED"

    def evaluate(verdict_fn, name):
        tv = [k for k in range(len(te)) if gold(*te[k][1:]) == "VERIFIED"]
        cov = fv = 0
        for k in range(len(te)):
            v = verdict_fn(te[k][0]); g = gold(*te[k][1:])
            if v == "VERIFIED" and g == "VERIFIED": cov += 1
            if v == "VERIFIED" and g != "VERIFIED": fv += 1
        print(f"  {name:22} VERIFIED coverage {cov}/{len(tv)} ({100*cov/max(1,len(tv)):.0f}%)  false-verify {fv}")
        return cov, fv, len(tv)

    print(f"test sentences: {len(te)}\n")
    r_cov, r_fv, ntv = evaluate(lambda s: cv.verify_claim(s).verdict, "rule-alone")
    p_cov, p_fv, _ = evaluate(pv_verdict, "propose-and-verify")
    print()
    gain = p_cov - r_cov
    g1 = p_fv <= r_fv                          # propose-and-verify is no less safe than rule-alone
    g2 = p_cov >= r_cov                         # and covers at least as much (ideally more)
    print(f"  coverage gain {gain:+d} VERIFIED, false-verify {r_fv}->{p_fv}")
    print(f"  [{'PASS' if g1 else 'FAIL'}] propose-and-verify no less safe than rule-alone (fv {p_fv}<={r_fv})")
    print(f"  [{'PASS' if g2 else 'FAIL'}] propose-and-verify covers >= rule-alone")
    ok = g1 and g2
    print(f"\n  >>> {'PASS — learned proposer + grounding expands safe coverage' if ok else 'CHECK — grounding too strict or proposer weak'}")
    print("\nHonest: bag-of-words proposer + grounding (entities-present + direction-cue) verifier. The")
    print("grounding is the independent exact check that keeps the red-line: we never VERIFY a claim")
    print("whose entities/direction aren't literally in the text. Coverage scales with proposer quality")
    print("(bert-base hits 94-95% extraction, NOTE-119) — the same grounding then admits more safely.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
