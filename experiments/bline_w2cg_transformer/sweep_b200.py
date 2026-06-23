"""B200 sweep — fine-tune a panel of pretrained LMs (distilbert / bert-base / bert-large) across
seeds as the W2CG proposer, the genuinely GPU-bound workload. Self-contained for the pod: reads
corpus.txt, imports the bundled claim_verifier.py for the rule-anchored red-line, writes results.json.

Reports per (model, seed): held-out extraction accuracy (cause/effect/dir) and the rule-anchored
hybrid red-line (VERIFIED requires the exact rule lookup, so the learned model can extend recall
or abstain but never false-verify). The panel + seeds make it a real GPU job, not a toy fit.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from claim_verifier import ClaimVerifier            # bundled alongside this file

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CAUSES = ["none", "smoking", "exercise", "drug", "alcohol", "sleep", "diet", "stress", "vaccine"]
EFFECTS = ["none", "cancer", "mortality", "recovery", "heart_disease", "diabetes", "depression", "infection"]
DIRS = ["+", "-", "0", "none"]
MODELS = os.environ.get("MODELS", "distilbert-base-uncased,bert-base-uncased,bert-large-uncased").split(",")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "8"))

SYN = {"smoking": ["smoking", "lighting up", "light up", "puffing", "cigarette", "cig", "tobacco", "smoke", "pack a day"],
       "alcohol": ["alcohol", "booz", "drink", "beer", "wine"],
       "exercise": ["exercise", "gym", "working out", "work out", "sweat", "jogging", "cardio"],
       "drug": ["drug", "pill", "medication", "medicine"], "vaccine": ["vaccine", "vaccin", "jab", "shot", "immuniz"],
       "sleep": ["sleep", "z's", "rest", "shut-eye"], "stress": ["stress", "anxiet", "burnout"],
       "diet": ["diet", "junk food", "veggies", "eating"], "cancer": ["cancer", "tumor", "tumour", "the big c"],
       "heart_disease": ["heart_disease", "ticker", "heart attack", "heart disease", "cardiac"],
       "depression": ["depression", "depress", "the blues", "feeling down"],
       "mortality": ["mortality", "six feet under", "die", "death", "live longer", "kick the bucket"],
       "diabetes": ["diabetes", "diabet", "blood sugar"], "infection": ["infection", "infect", "the flu", "a cold", "virus"],
       "recovery": ["recovery", "recover", "getting better", "heal"]}


def load():
    rows = []
    for ln in open(os.path.join(HERE, "corpus.txt")):
        p = [x.strip() for x in ln.split("|", 3)]
        if len(p) == 4 and p[0] in CAUSES and p[1] in EFFECTS and p[2] in DIRS:
            rows.append((p[3], p[0], p[1], p[2]))
    return rows


class LMExtractor(nn.Module):
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(name)
        d = self.enc.config.hidden_size
        self.hc, self.he, self.hd = nn.Linear(d, len(CAUSES)), nn.Linear(d, len(EFFECTS)), nn.Linear(d, len(DIRS))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.hc(h), self.he(h), self.hd(h)


def run_one(model_name, seed, tr, te, struct, cv):
    from transformers import AutoTokenizer
    torch.manual_seed(seed); np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)

    def enc(rows):
        b = tok([s for s, *_ in rows], padding=True, truncation=True, max_length=48, return_tensors="pt")
        return (b["input_ids"].to(DEVICE), b["attention_mask"].to(DEVICE),
                torch.tensor([CAUSES.index(c) for _, c, _, _ in rows], device=DEVICE),
                torch.tensor([EFFECTS.index(e) for _, _, e, _ in rows], device=DEVICE),
                torch.tensor([DIRS.index(d) for _, _, _, d in rows], device=DEVICE))

    Xi, Xm, yc, ye, yd = enc(tr); Ti, Tm, *_ = enc(te)
    model = LMExtractor(model_name).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-2)
    lf = nn.CrossEntropyLoss(); bs = 32
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(tr))
        for i in range(0, len(tr), bs):
            b = perm[i:i + bs]
            pc, pe, pd = model(Xi[b], Xm[b])
            loss = lf(pc, yc[b]) + lf(pe, ye[b]) + lf(pd, yd[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pc, pe, pd = model(Ti, Tm)
    ic, ie, id_ = pc.argmax(1).cpu().numpy(), pe.argmax(1).cpu().numpy(), pd.argmax(1).cpu().numpy()
    gc = [CAUSES.index(c) for _, c, _, _ in te]; ge = [EFFECTS.index(e) for _, _, e, _ in te]
    gd = [DIRS.index(d) for _, _, _, d in te]; n = len(te)
    cok = int(sum(a == b for a, b in zip(ic, gc))); eok = int(sum(a == b for a, b in zip(ie, ge)))
    dok = int(sum(a == b for a, b in zip(id_, gd)))

    def verdict(c, e, d):
        if c == "none" or e == "none" or d == "none" or (c, e) not in struct: return "UNVERIFIABLE"
        return "VERIFIED" if d == struct[(c, e)] else "CONTRADICTED"
    fv = 0
    for k in range(n):
        rv = cv.verify_claim(te[k][0]).verdict
        if rv == "VERIFIED" and verdict(*[te[k][j] for j in (1, 2, 3)]) != "VERIFIED":
            fv += 1
    return {"model": model_name, "seed": seed, "n_test": n,
            "cause_acc": cok / n, "effect_acc": eok / n, "dir_acc": dok / n,
            "extract_acc": (cok + eok + dok) / (3 * n), "rule_red_line_false_verify": fv}


def main():
    rows = load()
    idx = np.random.default_rng(0).permutation(len(rows))
    cut = int(0.85 * len(rows)); tr = [rows[i] for i in idx[:cut]]; te = [rows[i] for i in idx[cut:]]
    from collections import defaultdict, Counter
    edge = defaultdict(Counter)
    for _, c, e, d in tr:
        if c != "none" and e != "none" and d != "none": edge[(c, e)][d] += 1
    struct = {k: v.most_common(1)[0][0] for k, v in edge.items()}
    rstruct = {k: {"direction": {"+": 1, "-": -1, "0": 0}[d], "magnitude": None} for k, d in struct.items()}
    cv = ClaimVerifier(rstruct, SYN)
    print(f"device={DEVICE} corpus={len(rows)} train={len(tr)} test={len(te)} models={MODELS} seeds={SEEDS} epochs={EPOCHS}")
    results = []
    for m in MODELS:
        for s in SEEDS:
            t0 = time.time()
            try:
                r = run_one(m, s, tr, te, struct, cv); r["sec"] = round(time.time() - t0, 1)
                print(f"  {m} seed{s}: extract {100*r['extract_acc']:.0f}% "
                      f"(c{100*r['cause_acc']:.0f}/e{100*r['effect_acc']:.0f}/d{100*r['dir_acc']:.0f}) "
                      f"redline_fv={r['rule_red_line_false_verify']} {r['sec']}s")
                results.append(r)
            except Exception as ex:
                print(f"  {m} seed{s}: ERROR {type(ex).__name__}: {str(ex)[:120]}")
    # aggregate per model
    agg = {}
    for m in MODELS:
        rs = [r for r in results if r["model"] == m]
        if rs:
            agg[m] = {"extract_acc_mean": float(np.mean([r["extract_acc"] for r in rs])),
                      "extract_acc_std": float(np.std([r["extract_acc"] for r in rs])),
                      "red_line_fv_mean": float(np.mean([r["rule_red_line_false_verify"] for r in rs]))}
    out = {"device": DEVICE, "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else DEVICE,
           "corpus": len(rows), "runs": results, "aggregate": agg}
    json.dump(out, open(os.path.join(HERE, "results_b200.json"), "w"), indent=2)
    print("\nAGGREGATE:")
    for m, a in agg.items():
        print(f"  {m}: extract {100*a['extract_acc_mean']:.1f}±{100*a['extract_acc_std']:.1f}%  red-line fv {a['red_line_fv_mean']:.1f}")
    print("wrote results_b200.json")


if __name__ == "__main__":
    main()
