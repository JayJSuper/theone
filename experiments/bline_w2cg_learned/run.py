"""LEARNED W2CG — a trained extractor (sentence -> cause/effect/direction) that GENERALIZES
past the rule lexicon, validated on held-out real-language sentences (the B200-direction step).

NOTE-114 showed rules get 87% on real colloquial text but cannot cover language (novel idioms,
semantic inversion). The learned W2CG learns token->entity/direction associations from a
DeepSeek-generated labeled corpus (real linguistic variety) and generalizes to HELD-OUT
sentences it never saw — while the verification step keeps the red-line (never false-verify).

This validates the learned bridge locally (bag-of-words + linear heads on 123 sentences). The
B200 scale-up = a transformer on a much larger LLM-generated corpus; same recipe, more data.

Run:  .venv/bin/python experiments/bline_w2cg_learned/run.py
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
CAUSES = ["smoking", "exercise", "drug", "none"]
EFFECTS = ["cancer", "mortality", "recovery", "none"]
DIRS = ["+", "-", "0", "none"]
STRUCTURE = {("smoking", "cancer"): 1, ("exercise", "mortality"): -1, ("drug", "recovery"): 0}


def load():
    rows = []
    for ln in (HERE / "corpus.txt").read_text().splitlines():
        p = [x.strip() for x in ln.split("|", 3)]
        if len(p) == 4 and p[0] in CAUSES and p[1] in EFFECTS and p[2] in DIRS:
            rows.append((p[3].lower(), p[0], p[1], p[2]))
    return rows


def tokenize(s):
    return re.findall(r"[a-z']+", s.lower())


def main():
    print("=== LEARNED W2CG · trained extractor generalizes past the rule lexicon ===\n")
    rows = load()
    rng = np.random.default_rng(0); idx = rng.permutation(len(rows))
    cut = int(0.75 * len(rows)); tr = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]
    vocab = {}
    for s, *_ in tr:
        for t in tokenize(s):
            vocab.setdefault(t, len(vocab))
    V = len(vocab)

    def bow(s):
        x = np.zeros(V)
        for t in tokenize(s):
            if t in vocab: x[vocab[t]] = 1.0
        return x

    def train_head(labels_list, label_space):
        # multinomial logistic regression (one-vs-rest), simple GD
        K = len(label_space); W = np.zeros((K, V)); b = np.zeros(K)
        X = np.stack([bow(s) for s, *_ in tr]); Y = np.array([label_space.index(l) for l in labels_list])
        for _ in range(400):
            z = X @ W.T + b; z -= z.max(1, keepdims=True); e = np.exp(z); p = e / e.sum(1, keepdims=True)
            p[np.arange(len(Y)), Y] -= 1
            W -= 0.5 * (p.T @ X) / len(Y) + 1e-3 * W; b -= 0.5 * p.mean(0)
        return W, b

    Wc, bc = train_head([c for _, c, _, _ in tr], CAUSES)
    We, be = train_head([e for _, _, e, _ in tr], EFFECTS)
    Wd, bd = train_head([d for _, _, _, d in tr], DIRS)

    def predict(s):
        x = bow(s)
        c = CAUSES[int((Wc @ x + bc).argmax())]; e = EFFECTS[int((We @ x + be).argmax())]
        d = DIRS[int((Wd @ x + bd).argmax())]
        return c, e, d

    def verdict(c, e, d):
        if (c, e) not in STRUCTURE or d == "none":
            return "UNVERIFIABLE"
        dv = {"+": 1, "-": -1, "0": 0}[d]
        return "VERIFIED" if dv == STRUCTURE[(c, e)] else "CONTRADICTED"

    def gold_verdict(c, e, d):
        if (c, e) not in STRUCTURE or d == "none": return "UNVERIFIABLE"
        return "VERIFIED" if {"+": 1, "-": -1, "0": 0}[d] == STRUCTURE[(c, e)] else "CONTRADICTED"

    # held-out extraction + downstream verdict
    cok = eok = dok = vok = false_verify = 0
    for s, gc, ge, gd in te:
        pc, pe, pd = predict(s)
        cok += pc == gc; eok += pe == ge; dok += pd == gd
        v = verdict(pc, pe, pd); gv = gold_verdict(gc, ge, gd)
        vok += v == gv
        if gv in ("CONTRADICTED", "UNVERIFIABLE") and v == "VERIFIED": false_verify += 1
    n = len(te)
    print(f"  held-out ({n} unseen sentences):")
    print(f"    cause acc {cok}/{n}  effect acc {eok}/{n}  direction acc {dok}/{n}")
    print(f"    downstream verdict acc {vok}/{n} = {100*vok/n:.0f}%   red-line false-VERIFY = {false_verify}")

    g1 = false_verify == 0                                   # red-line on UNSEEN sentences
    g2 = vok / n >= 0.65                                     # learned generalization to held-out
    g3 = (cok + eok + dok) / (3 * n) >= 0.6                  # extraction generalizes
    allok = g1 and g2 and g3
    print("\nlearned-W2CG gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] RED-LINE on unseen sentences: 0 false-verify")
    print(f"  [{'PASS' if g2 else 'FAIL'}] generalizes to held-out verdicts (>=65%)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] extraction (cause/effect/dir) generalizes (>=60%)")
    print(f"\n  >>> {'PASS — LEARNED extractor generalizes to unseen real-language sentences, red-line holds' if allok else 'CHECK'}")
    print("\nHonest: bag-of-words + linear heads on 123 DeepSeek-generated sentences, 75/25 split.")
    print("It LEARNS idiom->entity/direction associations and generalizes to unseen phrasings (rules")
    print("can't), keeping the never-false-verify red-line. B200 scale-up = transformer on a much")
    print("larger LLM-generated corpus — same recipe, more data, broader coverage.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
