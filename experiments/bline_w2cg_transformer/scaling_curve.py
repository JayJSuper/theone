"""Scaling-curve test of the B200 hypothesis — does the Transformer W2CG extractor IMPROVE with
corpus size, toward/past the bag-of-words baseline? A rising curve justifies spending B200 to
scale the corpus to thousands; a flat curve says scale won't help (pivot before burning GPU).

Trains the same Transformer on increasing fractions of the train split and reports held-out
extraction accuracy vs the bag-of-words baseline at each size. CPU/MPS/CUDA agnostic.

Run:  .venv/bin/python experiments/bline_w2cg_transformer/scaling_curve.py
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn
from run import (load, tok, encode, W2CGTransformer, _bow_baseline,
                 CAUSES, EFFECTS, DIRS, PAD, DEVICE)

FAST = os.environ.get("THEONE_FAST") == "1"


def train_eval(tr, te, vocab, epochs):
    def batch(rows):
        X = torch.tensor([encode(s, vocab) for s, *_ in rows], dtype=torch.long, device=DEVICE)
        yc = torch.tensor([CAUSES.index(c) for _, c, _, _ in rows], device=DEVICE)
        ye = torch.tensor([EFFECTS.index(e) for _, _, e, _ in rows], device=DEVICE)
        yd = torch.tensor([DIRS.index(d) for _, _, _, d in rows], device=DEVICE)
        return X, yc, ye, yd
    Xtr, yc, ye, yd = batch(tr); Xte, *_ = batch(te)
    torch.manual_seed(0)
    model = W2CGTransformer(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    lf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 32):
            b = perm[i:i + 32]
            pc, pe, pd = model(Xtr[b])
            loss = lf(pc, yc[b]) + lf(pe, ye[b]) + lf(pd, yd[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pc, pe, pd = model(Xte)
    ic, ie, id_ = pc.argmax(1).cpu().numpy(), pe.argmax(1).cpu().numpy(), pd.argmax(1).cpu().numpy()
    gc = [CAUSES.index(c) for _, c, _, _ in te]; ge = [EFFECTS.index(e) for _, _, e, _ in te]
    gd = [DIRS.index(d) for _, _, _, d in te]; n = len(te)
    acc = (sum(a == b for a, b in zip(ic, gc)) + sum(a == b for a, b in zip(ie, ge))
           + sum(a == b for a, b in zip(id_, gd))) / (3 * n)
    return acc


def main():
    rows = load()
    idx = np.random.default_rng(0).permutation(len(rows))
    cut = int(0.80 * len(rows)); trall = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]
    epochs = 40 if FAST else 120
    print(f"=== W2CG scaling curve · device={DEVICE} · total={len(rows)} test={len(te)} fast={FAST} ===\n")
    print(f"  {'train_n':>8} {'transformer':>12} {'bag-of-words':>13}")
    fracs = [0.25, 0.5, 0.75, 1.0]
    tf_accs = []
    for f in fracs:
        k = max(8, int(f * len(trall))); tr = trall[:k]
        vocab = {"<pad>": PAD, "<unk>": 1}
        for s, *_ in tr:
            for t in tok(s):
                vocab.setdefault(t, len(vocab))
        tf = train_eval(tr, te, vocab, epochs)
        bw = _bow_baseline(tr, te)
        tf_accs.append(tf)
        print(f"  {k:>8} {100*tf:>11.0f}% {100*bw:>12.0f}%")
    rising = tf_accs[-1] > tf_accs[0] + 0.03                 # transformer improves with data
    print(f"\n  transformer {100*tf_accs[0]:.0f}% -> {100*tf_accs[-1]:.0f}% as train grows "
          f"{int(0.25*len(trall))}->{len(trall)}")
    print(f"  >>> {'RISING — more data helps; B200 corpus-scaling is justified' if rising else 'FLAT — scale alone wont help; pivot (richer model/features/propose-verify)'}")
    print("\nHonest: small absolute sizes (hundreds); the SHAPE of the curve is the signal for whether")
    print("thousands-scale B200 training pays off. Red-line stays anchored by the rule verifier either way.")


if __name__ == "__main__":
    main()
