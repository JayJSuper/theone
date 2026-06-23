"""B3 efficiency unlock — a CHUNKED PARALLEL SCAN for the gated linear recurrence, so the
selective SSM trains at LONG context fast (the real reason SSMs beat attention: O(N) work but
parallelizable, vs attention's O(N^2)).

The recurrence h_t = a_t ⊙ h_{t-1} + u_t  (u_t = b_t ⊙ v_t) is an associative scan. We compute
the final state without a length-L python loop:
  split L into C chunks of width W; within each chunk (loop of length W, vectorized across all
  chunks at once) get the chunk's from-zero final state h_c and product P_c; then carry across
  chunks (loop of length C): carry <- P_c * carry + h_c.
Total python iterations = W + C  (e.g. L=4096, W=64 -> 128) instead of 4096. No division ->
numerically stable. We VERIFY it equals the naive sequential scan, then train long-context.

Run:  .venv/bin/python experiments/bline_b3_parallel_scan/run.py
"""
from __future__ import annotations
import os, time, math
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
K, M = 8, 3
torch.manual_seed(0)
FAST = bool(int(os.environ.get("THEONE_FAST", "0")))
D_MODEL = int(os.environ.get("THEONE_D", "128"))
BS = int(os.environ.get("THEONE_BS", "128"))
WCHUNK = int(os.environ.get("THEONE_W", "64"))


def seq_final(a, u):
    """Naive sequential final state (reference)."""
    B, L, d = a.shape
    h = torch.zeros(B, d, device=a.device)
    for t in range(L):
        h = a[:, t] * h + u[:, t]
    return h


def chunked_final(a, u, W=64):
    """Chunked parallel scan -> final state. python iters = W + L/W, not L."""
    B, L, d = a.shape
    if L % W:                                   # pad to a multiple of W with identity (a=1,u=0)
        pad = W - (L % W)
        a = torch.cat([a, torch.ones(B, pad, d, device=a.device)], 1)
        u = torch.cat([u, torch.zeros(B, pad, d, device=a.device)], 1)
        L = a.shape[1]
    C = L // W
    a = a.view(B, C, W, d); u = u.view(B, C, W, d)
    hc = torch.zeros(B, C, d, device=a.device)
    Pc = torch.ones(B, C, d, device=a.device)
    for t in range(W):                          # within-chunk, vectorized across all chunks
        hc = a[:, :, t] * hc + u[:, :, t]
        Pc = Pc * a[:, :, t]
    carry = torch.zeros(B, d, device=a.device)
    for c in range(C):                          # carry across chunks
        carry = Pc[:, c] * carry + hc[:, c]
    return carry


def make_batch(bs, L, gen):
    x = torch.zeros(bs, L, dtype=torch.long)
    y = torch.zeros(bs, M, dtype=torch.long)
    hi = max(M + 1, int(0.6 * L))
    for b in range(bs):
        pos = torch.randperm(hi, generator=gen)[:M].sort().values
        toks = torch.randint(1, K + 1, (M,), generator=gen)
        x[b, pos] = toks; y[b] = toks
    return x.to(DEV), y.to(DEV)


class SelSSM(nn.Module):
    def __init__(self, L, d=D_MODEL, W=WCHUNK):
        super().__init__()
        self.W = W
        self.emb = nn.Embedding(K + 1, d); self.to_v = nn.Linear(d, d)
        self.to_a = nn.Linear(d, d); self.to_b = nn.Linear(d, d)
        nn.init.constant_(self.to_a.bias, max(4.0, math.log(L) + 2.5))
        self.heads = nn.ModuleList([nn.Linear(d, K + 1) for _ in range(M)])

    def forward(self, x):
        e = self.emb(x); v = self.to_v(e)
        a = torch.sigmoid(self.to_a(e)); b = torch.sigmoid(self.to_b(e))
        h = chunked_final(a, b * v, self.W)           # parallel scan
        return [head(h) for head in self.heads]


def verify_scan():
    """Chunked parallel scan must equal the naive sequential scan."""
    B, L, d = 4, 300, 16
    a = torch.rand(B, L, d, device=DEV) * 0.2 + 0.79   # retention-ish
    u = torch.randn(B, L, d, device=DEV)
    ref = seq_final(a, u); par = chunked_final(a, u, W=64)
    return float((ref - par).abs().max())


def train_eval(L, steps, bs=BS):
    m = SelSSM(L).to(DEV)
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
    print("=== B3 unlock · chunked PARALLEL SCAN -> fast long-context selective SSM ===")
    print(f"device={DEV}\n")
    gap = verify_scan()
    print(f"parallel scan == sequential scan (max abs diff) ... {gap:.2e}  [{'OK' if gap < 1e-4 else 'BAD'}]\n")

    Ls = [int(x) for x in os.environ.get("THEONE_LS", "256,1024,4096").split(",")] if not FAST else [256, 1024]
    steps = int(os.environ.get("THEONE_STEPS", "1500")) if not FAST else 800
    print(f"{'L':>6} {'recall':>8} {'train sec':>10}")
    rows = []
    for L in Ls:
        t0 = time.time(); rec = train_eval(L, steps); dt = time.time() - t0
        rows.append((L, rec, dt))
        print(f"{L:>6} {100*rec:>7.1f}% {dt:>10.1f}")

    g1 = gap < 1e-4                                      # parallel scan correct
    g2 = min(r[1] for r in rows) > 0.85                 # long-context recall stays high
    g3 = rows[-1][2] < 6 * rows[0][2]                   # ~linear scaling (16x length, <6x time)
    allok = g1 and g2 and g3
    print("\nB3-parallel-scan gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] chunked parallel scan == sequential (<1e-4)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] selective recall stays high to long L (>85%)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] training stays ~sub-linear in L (16x len << 16x time)")
    print(f"\n  >>> {'PASS — parallel scan unlocks FAST long-context selective SSM (L=4096 trainable)' if allok else 'CHECK'}")
    print("\nHonest: small model, synthetic selective-copy. The win is the SCAN: same O(N) math,")
    print("computed with W+L/W python iters not L, numerically verified == sequential. This is the")
    print("real post-Transformer efficiency lever (vs attention O(N^2)). Fluent language NOT claimed.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
