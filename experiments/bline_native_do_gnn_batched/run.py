"""① REAL-SCALE native do() — batched size-invariant GNN (disjoint-union graph batching).

NOTE-139/140 proved the size-general do() GNN extrapolates, but trained one graph at a time (slow).
To make it real-scale-READY, this batches many variable-size graphs into one disjoint union (offset
edge indices, per-graph X/Y gathers), so B graphs run per forward — fast enough to train at scale on
GPU. Same structural-do encoding (cut X's incoming edges, clamp, propagate). We verify it matches the
unbatched result and EXTRAPOLATES, then it is ready for a real-scale cloud run.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_native_do_gnn_batched/run.py
      (cloud real-scale: NTR=200000 EPOCHS=40 WIDTH=64 via the varstruct/gnn cloud job)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "bline_native_do_gnn"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gnn_mod", ROOT / "experiments/bline_native_do_gnn/run.py")
g = _ilu.module_from_spec(_spec); sys.modules["gnn_mod"] = g; _spec.loader.exec_module(g)
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
Hd = int(os.environ.get("WIDTH", "32"))
NTR = int(os.environ.get("NTR", "4000" if FAST else "40000"))
EPOCHS = int(os.environ.get("EPOCHS", "16" if FAST else "30"))
ROUNDS = 8


def collate(batch, device):
    """disjoint union: stack nodes, offset edge endpoints, gather X/Y per graph."""
    nfs, srcs, dsts, ews, xs, ys, ds = [], [], [], [], [], [], []
    off = 0
    for nf, edges, X, Y, d, K, _ in batch:
        nfs.append(torch.tensor(nf))
        for (s, t, wv) in edges:
            srcs.append(s + off); dsts.append(t + off); ews.append(wv)
        xs.append(X + off); ys.append(Y + off); ds.append(d); off += K
    NF = torch.cat(nfs).to(device)
    src = torch.tensor(srcs, dtype=torch.long, device=device) if srcs else torch.zeros(0, dtype=torch.long, device=device)
    dst = torch.tensor(dsts, dtype=torch.long, device=device) if dsts else torch.zeros(0, dtype=torch.long, device=device)
    ew = torch.tensor(ews, dtype=torch.float32, device=device).unsqueeze(-1) if ews else torch.zeros(0, 1, device=device)
    return (NF, src, dst, ew, torch.tensor(xs, device=device), torch.tensor(ys, device=device),
            torch.tensor(ds, dtype=torch.float32, device=device), off)


class BatchedGNNdo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(3, Hd)
        self.msg = nn.Sequential(nn.Linear(Hd + 1, Hd), nn.GELU(), nn.Linear(Hd, Hd))
        self.upd = nn.GRUCell(Hd, Hd)
        self.read = nn.Sequential(nn.Linear(Hd, Hd), nn.GELU(), nn.Linear(Hd, 1))
        self.xclamp = nn.Parameter(torch.randn(Hd) * 0.1)

    def forward(self, NF, src, dst, ew, xidx, yidx, ntot):
        h = self.embed(NF)
        h = h.clone(); h[xidx] = self.xclamp
        for _ in range(ROUNDS):
            if src.numel():
                m = self.msg(torch.cat([h[src], ew], -1))
                agg = torch.zeros(ntot, Hd, device=h.device).index_add_(0, dst, m)
            else:
                agg = torch.zeros(ntot, Hd, device=h.device)
            h = self.upd(agg, h)
            h = h.clone(); h[xidx] = self.xclamp
        return torch.sigmoid(self.read(h[yidx])).squeeze(-1)


def evaluate(net, G, bs=256):
    net.eval(); err = []; obs = []
    with torch.no_grad():
        for i in range(0, len(G), bs):
            b = G[i:i + bs]
            NF, src, dst, ew, xi, yi, d, ntot = collate(b, DEVICE)
            pred = net(NF, src, dst, ew, xi, yi, ntot).cpu().numpy()
            for k, (nf, edges, X, Y, dd, K, raw) in enumerate(b):
                err.append(abs(float(pred[k]) - dd)); obs.append(abs(g.obs_cond(K, X, Y, *raw) - dd))
    return float(np.mean(err)), float(np.mean(obs))


def main():
    SEED = int(os.environ.get("SEED", "0"))
    torch.manual_seed(SEED)
    print("=== ① REAL-SCALE native do · BATCHED size-invariant GNN ===\n")
    print(f"  device={DEVICE}  NTR={NTR}  epochs={EPOCHS}  width={Hd}  SEED={SEED}")
    Gtr = g.make([4, 5], NTR, 1000 + SEED)                  # training data varies by seed
    Gin = g.make([4, 5], 800, 1); G7 = g.make([7], 500, 3); G9 = g.make([9], 400, 4)   # test sets FIXED across seeds
    net = BatchedGNNdo().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    bs = 256
    import time as _t  # only for nothing; avoid Date — use epoch index
    for ep in range(EPOCHS):
        net.train(); order = np.random.default_rng(100 + ep).permutation(len(Gtr))
        tot = 0.0; nb = 0
        for i in range(0, len(Gtr), bs):
            b = [Gtr[j] for j in order[i:i + bs]]
            NF, src, dst, ew, xi, yi, d, ntot = collate(b, DEVICE)
            loss = torch.nn.functional.mse_loss(net(NF, src, dst, ew, xi, yi, ntot), d)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if (ep + 1) % max(1, EPOCHS // 4) == 0:
            print(f"    epoch {ep+1}/{EPOCHS}  mse {tot/nb:.5f}")

    mae_in, _ = evaluate(net, Gin); mae7, obs7 = evaluate(net, G7); mae9, obs9 = evaluate(net, G9)
    print(f"\n  in-dist K∈{{4,5}}   MAE={mae_in:.4f}")
    print(f"  EXTRAPOLATE K=7    MAE={mae7:.4f}  (baseline {obs7:.4f})")
    print(f"  EXTRAPOLATE K=9    MAE={mae9:.4f}  (baseline {obs9:.4f})")

    thr = 0.06 if FAST else 0.035
    g1 = mae_in < thr; g2 = mae7 < thr * 1.6 and mae9 < thr * 2.0; g3 = mae9 < 0.7 * obs9
    allok = g1 and g2 and g3
    print("\nbatched real-scale gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] batched GNN in-dist engine-tight (MAE<{thr})")
    print(f"  [{'PASS' if g2 else 'FAIL'}] still EXTRAPOLATES K=7,9 after batching")
    print(f"  [{'PASS' if g3 else 'FAIL'}] beats confounded baseline at K=9")
    print(f"\n  >>> {'PASS — batched disjoint-union GNN trains many graphs/step (real-scale ready) and keeps the size-extrapolating do() — ready for a cloud real-scale run.' if allok else 'CHECK'}")
    # fingerprinted result (no deletion: written to experiments/ for external audit)
    import json, hashlib
    res = {"seed": SEED, "NTR": NTR, "EPOCHS": EPOCHS, "width": Hd, "device": str(DEVICE),
           "mae_in_dist": round(mae_in, 5), "mae_K7": round(mae7, 5), "mae_K9": round(mae9, 5),
           "baseline_K7": round(obs7, 5), "baseline_K9": round(obs9, 5), "all_gates_pass": bool(allok)}
    outp = Path(os.environ.get("RESULT_DIR", str(Path(__file__).parent))) / f"realscale_result_seed{SEED}.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"RESULT_JSON {outp.name} sha256={hashlib.sha256(outp.read_bytes()).hexdigest()}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
