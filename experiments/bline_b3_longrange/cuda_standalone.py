"""Self-contained CUDA B3 long-range — high capacity, long L, on a RunPod GPU. Zero repo deps
(torch+numpy only), so it bootstraps on a bare PyTorch pod. Trains the selective SSM and the
content-agnostic reservoir at growing sequence lengths and prints a parseable RESULT per L.

Goal: with enough capacity (which the GPU makes affordable at long L) selective recall stays
HIGH as L grows, while the reservoir stays near chance — the length-robust selection the local
MPS run was too slow to train. Env THEONE_LS overrides the L sweep.
"""
import os, time, math
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K, M = 8, 3
torch.manual_seed(0)
LS = [int(x) for x in os.environ.get("THEONE_LS", "128,256,512,1024").split(",")]
D = int(os.environ.get("THEONE_D", "128"))
STEPS = int(os.environ.get("THEONE_STEPS", "3000"))
BS = int(os.environ.get("THEONE_BS", "128"))


def make_batch(bs, L, gen):
    x = torch.zeros(bs, L, dtype=torch.long)
    y = torch.zeros(bs, M, dtype=torch.long)
    hi = max(M + 1, int(0.6 * L))
    for b in range(bs):
        pos = torch.randperm(hi, generator=gen)[:M].sort().values
        toks = torch.randint(1, K + 1, (M,), generator=gen)
        x[b, pos] = toks; y[b] = toks
    return x.to(DEV), y.to(DEV)


class SSM(nn.Module):
    def __init__(self, L, d=D, selective=True):
        super().__init__()
        self.selective = selective; self.d = d
        self.emb = nn.Embedding(K + 1, d); self.to_v = nn.Linear(d, d)
        ret = max(4.0, math.log(L) + 2.5)
        if selective:
            self.to_a = nn.Linear(d, d); self.to_b = nn.Linear(d, d)
            nn.init.constant_(self.to_a.bias, ret)
        else:
            self.log_a = nn.Parameter(torch.full((d,), ret)); self.log_b = nn.Parameter(torch.zeros(d))
        self.heads = nn.ModuleList([nn.Linear(d, K + 1) for _ in range(M)])

    def forward(self, x):
        e = self.emb(x); B, L, d = e.shape; v = self.to_v(e)
        if self.selective:
            a = torch.sigmoid(self.to_a(e)); b = torch.sigmoid(self.to_b(e))
        else:
            a = torch.sigmoid(self.log_a).view(1, 1, d).expand(B, L, d)
            b = torch.sigmoid(self.log_b).view(1, 1, d).expand(B, L, d)
        h = torch.zeros(B, d, device=x.device)
        for t in range(L):
            h = a[:, t] * h + b[:, t] * v[:, t]
        return [head(h) for head in self.heads]


def train_eval(selective, L, steps, bs=BS):
    m = SSM(L, selective=selective).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); gen = torch.Generator().manual_seed(0)
    lossf = nn.CrossEntropyLoss()
    for s in range(steps):
        x, y = make_batch(bs, L, gen)
        lo = m(x); loss = sum(lossf(lo[i], y[:, i]) for i in range(M))
        opt.zero_grad(); loss.backward(); opt.step()
    xe, ye = make_batch(400, L, torch.Generator().manual_seed(7))
    with torch.no_grad():
        lo = m(xe)
        per = torch.stack([lo[i].argmax(1) == ye[:, i] for i in range(M)], 1).float().mean().item()
    return per


def main():
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"THEONE_B3LR device={DEV} gpu={name} d={D} steps={STEPS} Ls={LS}", flush=True)
    for L in LS:
        t0 = time.time()
        sel = train_eval(True, L, STEPS)
        lti = train_eval(False, L, STEPS)
        print(f"RESULT L={L} selective={sel:.4f} reservoir={lti:.4f} gap={sel-lti:.4f} sec={time.time()-t0:.1f}",
              flush=True)
    print("THEONE_B3LR_DONE", flush=True)


if __name__ == "__main__":
    main()
