"""Learned fluent realizer — a trained model that renders a VERIFIED structured claim as fluent
natural language, round-trip-gated so it can only emit what re-parses to exactly that claim.

The generation counterpart to the learned W2CG proposer. NOTE-126 solved the architecture
(decouple truth from surface; round-trip gate); that used hand templates. Here the surface
realizer is LEARNED: fine-tune T5 to map (cause,effect,direction) -> a fluent sentence, training
on the W2CG corpus REVERSED (each labeled sentence is a structure->sentence pair). At inference we
sample several renderings and keep only those that round-trip via the W2CG verifier to the input
claim — fluency learned, truth still guaranteed by the gate.

Local smoke:  THEONE_FAST=1 .venv/bin/python experiments/bline_b2_learned_realizer/run.py
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
FAST = os.environ.get("THEONE_FAST") == "1"
MODEL = os.environ.get("MODEL", "t5-small")
EPOCHS = int(os.environ.get("EPOCHS", "2" if FAST else "5"))
DEVICE = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# entity synonyms for the W2CG round-trip verifier (broad schema)
SYN = {"smoking": ["smoking", "lighting up", "puffing", "cigarette", "cig", "tobacco", "smoke"],
       "alcohol": ["alcohol", "booz", "drink", "beer", "wine"], "exercise": ["exercise", "gym", "working out", "work out", "sweat", "cardio"],
       "drug": ["drug", "pill", "medication", "medicine"], "vaccine": ["vaccine", "vaccin", "jab", "shot", "immuniz"],
       "sleep": ["sleep", "z's", "rest"], "stress": ["stress", "anxiet", "burnout"], "diet": ["diet", "junk food", "veggies", "eating"],
       "cancer": ["cancer", "tumor", "tumour"], "heart_disease": ["heart_disease", "ticker", "heart attack", "heart disease", "cardiac", "heart"],
       "depression": ["depression", "depress", "the blues"], "mortality": ["mortality", "die", "death", "live longer", "kick the bucket"],
       "diabetes": ["diabetes", "diabet", "blood sugar"], "infection": ["infection", "infect", "the flu", "cold", "virus"],
       "recovery": ["recovery", "recover", "getting better", "heal"]}
DIRW = {"+": "increases", "-": "decreases", "0": "does not affect"}


def load_pairs():
    pairs = []
    for ln in (ROOT / "experiments/bline_w2cg_transformer/corpus.txt").read_text().splitlines():
        p = [x.strip() for x in ln.split("|", 3)]
        if len(p) == 4 and p[0] != "none" and p[1] != "none" and p[2] in ("+", "-", "0"):
            src = f"render causal claim: cause={p[0]} effect={p[1]} direction={DIRW[p[2]]}"
            pairs.append((src, p[3], p[0], p[1], p[2]))
    return pairs


def main():
    from transformers import T5ForConditionalGeneration, T5TokenizerFast
    from theone.language import ClaimVerifier
    torch.manual_seed(0)
    pairs = load_pairs()
    idx = np.random.default_rng(0).permutation(len(pairs))
    if FAST:
        idx = idx[:1200]
    cut = int(0.9 * len(idx)); tr = [pairs[i] for i in idx[:cut]]; te = [pairs[i] for i in idx[cut:]]
    print(f"model={MODEL} device={DEVICE} train={len(tr)} test={len(te)} epochs={EPOCHS} fast={FAST}")

    tok = T5TokenizerFast.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    bs = 16
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = [tr[j] for j in perm[i:i + bs].tolist()]
            enc = tok([x[0] for x in b], padding=True, truncation=True, max_length=48, return_tensors="pt").to(DEVICE)
            lab = tok([x[1] for x in b], padding=True, truncation=True, max_length=48, return_tensors="pt").input_ids.to(DEVICE)
            lab[lab == tok.pad_token_id] = -100
            loss = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, labels=lab).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  epoch {ep+1}/{EPOCHS} loss {loss.item():.3f}")

    # structure used by the round-trip verifier (the held-out claims' edges)
    struct = {}
    for _, _, c, e, d in pairs:
        struct.setdefault((c, e), {"direction": {"+": 1, "-": -1, "0": 0}[d], "magnitude": None})
    cv = ClaimVerifier(struct, SYN)

    model.eval()
    emitted = roundtrip_ok = distinct = 0; shown = 0
    for src, gold, c, e, d in te:
        enc = tok(src, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outs = model.generate(**enc, do_sample=True, top_p=0.92, num_return_sequences=4,
                                  max_new_tokens=40, num_beams=1)
        renders = list({tok.decode(o, skip_special_tokens=True) for o in outs})
        want = {"direction": {"+": 1, "-": -1, "0": 0}[d]}
        kept = []
        for r in renders:
            v = cv.verify_claim(r)
            if v.verdict == "VERIFIED" and v.cause == c and v.effect == e and v.direction == want["direction"]:
                kept.append(r)
        emitted += 1; roundtrip_ok += (len(kept) > 0); distinct += len(set(kept))
        if shown < 6 and kept:
            print(f"    [{c}->{e} {d}] {kept[0][:90]}"); shown += 1

    n = len(te)
    rate = roundtrip_ok / n
    print(f"\n  held-out structures ({n}):  >=1 round-trip-faithful rendering for {roundtrip_ok}/{n} ({100*rate:.0f}%)")
    print(f"  avg distinct faithful renderings/claim = {distinct/max(1,n):.2f}")
    g1 = rate >= 0.55                                        # learned realizer produces faithful fluent text
    print("\nlearned-realizer gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] learned T5 renders faithful (round-trip) fluent text for >=55% of held-out claims")
    print(f"\n  >>> {'PASS — learned fluent realizer: surface wording LEARNED, truth guaranteed by the round-trip gate' if g1 else 'CHECK — needs more epochs/data'}")
    print("\nHonest: T5 learns the surface realization from the corpus; the round-trip W2CG gate keeps")
    print("the red-line (only renderings that re-parse to the exact claim are emitted). Scale-up =")
    print("t5-base/large on the full corpus (cloud) for higher fidelity + diversity.")
    if not g1:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
