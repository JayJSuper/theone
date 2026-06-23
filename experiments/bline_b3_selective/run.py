"""B3 advance — a SELECTIVE O(N) state-space model does long-range selective recall (what
attention is used for) at linear cost, where a content-agnostic linear SSM cannot.

The existing SSM (ssm_encoder) is a fixed echo-state reservoir: O(N) and reconstructs a
continuous signal, but its dynamics are input-INDEPENDENT, so it cannot SELECT which past
content to remember. Real cognition needs content-selective long-range memory. This builds
the Mamba/S6 core idea: an input-dependent diagonal recurrence
    h_t = a_t ⊙ h_{t-1} + b_t ⊙ v_t,   a_t,b_t,v_t = fn(x_t)   (a_t = per-channel forget gate)
which is still O(N) (one sequential scan), but can choose to write content tokens into state
and preserve them across long blank spans.

Task — selective copy: in a length-L sequence of mostly BLANK tokens, M content tokens sit at
random early positions; the model must output those M tokens IN ORDER from its final state.
Requires carrying specific content across the whole sequence. We show:
  1. the selective SSM solves it at long L;
  2. a NON-selective (input-independent / LTI) SSM of equal size cannot (it can't select);
  3. forward cost is O(N) in L (linear), the asymptotic win over attention's O(N^2).

Run:  .venv/bin/python experiments/bline_b3_selective/run.py
"""
from __future__ import annotations
import os
import time
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available()
                   else "mps" if torch.backends.mps.is_available() else "cpu")
K = 8          # content vocab size (tokens 1..K); 0 = blank
M = 3          # number of content tokens to recall, in order
torch.manual_seed(0)


def make_batch(bs, L, gen):
    """length-L sequences: M content tokens (1..K) at random positions in the first 60%,
    rest blank (0). Target = the M content tokens in order of appearance."""
    x = torch.zeros(bs, L, dtype=torch.long)
    y = torch.zeros(bs, M, dtype=torch.long)
    hi = max(M + 1, int(0.6 * L))
    for b in range(bs):
        pos = torch.randperm(hi, generator=gen)[:M].sort().values
        toks = torch.randint(1, K + 1, (M,), generator=gen)
        x[b, pos] = toks
        y[b] = toks
    return x.to(DEV), y.to(DEV)


class SSMBlock(nn.Module):
    """One diagonal recurrence layer; selective (input-dependent a,b) or LTI (fixed a,b).
    Returns the FULL hidden sequence so blocks can stack. O(N) sequential scan."""
    def __init__(self, d, selective):
        super().__init__()
        self.d = d; self.selective = selective
        self.to_v = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        if selective:
            self.to_a = nn.Linear(d, d); self.to_b = nn.Linear(d, d)
            nn.init.constant_(self.to_a.bias, 4.0)      # retention init (sigmoid(4)=0.98)
        else:
            self.log_a = nn.Parameter(torch.full((d,), 4.0))
            self.log_b = nn.Parameter(torch.zeros(d))

    def forward(self, e):
        B, L, d = e.shape
        v = self.to_v(e)
        if self.selective:
            a = torch.sigmoid(self.to_a(e)); b = torch.sigmoid(self.to_b(e))
        else:
            a = torch.sigmoid(self.log_a).view(1, 1, d).expand(B, L, d)
            b = torch.sigmoid(self.log_b).view(1, 1, d).expand(B, L, d)
        h = torch.zeros(B, d, device=e.device); hs = []
        for t in range(L):
            h = a[:, t] * h + b[:, t] * v[:, t]
            hs.append(h)
        H = torch.stack(hs, dim=1)                       # (B, L, d)
        return e + self.out(torch.nn.functional.silu(H))  # residual


class SelectiveSSM(nn.Module):
    """Stacked diagonal-SSM blocks; final-state readout into M ordered classifiers."""
    def __init__(self, d=96, selective=True, layers=2):
        super().__init__()
        self.emb = nn.Embedding(K + 1, d)
        self.blocks = nn.ModuleList([SSMBlock(d, selective) for _ in range(layers)])
        self.heads = nn.ModuleList([nn.Linear(d, K + 1) for _ in range(M)])

    def forward(self, x):
        e = self.emb(x)
        for blk in self.blocks:
            e = blk(e)                                   # O(N) per layer
        h = e[:, -1]                                     # final state
        return [head(h) for head in self.heads]


FAST = bool(int(os.environ.get("THEONE_FAST", "0")))   # dashboard smoke mode: shorter & quicker


def train_eval(selective, L, steps=2500, bs=128):
    if FAST:
        steps = 1200
    m = SelectiveSSM(d=128, selective=selective, layers=2).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    lossf = nn.CrossEntropyLoss()
    for s in range(steps):
        x, y = make_batch(bs, L, gen)
        logits = m(x)
        loss = sum(lossf(logits[i], y[:, i]) for i in range(M))
        opt.zero_grad(); loss.backward(); opt.step()
    # eval
    xe, ye = make_batch(512, L, torch.Generator().manual_seed(99))
    with torch.no_grad():
        lo = m(xe)
        correct = torch.stack([lo[i].argmax(1) == ye[:, i] for i in range(M)], 1)
        per_tok = correct.float().mean().item()
        seq = correct.all(1).float().mean().item()
    return per_tok, seq, m


def main():
    print("=== B3 advance · SELECTIVE O(N) SSM does long-range selective recall ===")
    print(f"device={DEV}  vocab={K}  recall M={M} tokens in order\n")

    L = 64 if FAST else 128
    print(f"selective-copy task at L={L} (content in first 60%, recall from final state):")
    sel_tok, sel_seq, _ = train_eval(True, L)
    lti_tok, lti_seq, _ = train_eval(False, L)
    chance = 1.0 / K
    print(f"  SELECTIVE SSM ... per-token {100*sel_tok:5.1f}%  whole-sequence {100*sel_seq:5.1f}%")
    print(f"  LTI (no select). per-token {100*lti_tok:5.1f}%  whole-sequence {100*lti_seq:5.1f}%  (chance {100*chance:.0f}%)")

    # O(N) scaling: forward wall-clock vs L should be ~linear
    print("\n  O(N) scaling (forward wall-clock vs sequence length):")
    m = SelectiveSSM(selective=True).to(DEV)
    Ls = [64, 128, 256, 512]; times = []
    for Lx in Ls:
        x, _ = make_batch(64, Lx, torch.Generator().manual_seed(1))
        with torch.no_grad():
            _ = m(x)                              # warmup
            t0 = time.time()
            for _ in range(3):
                _ = m(x)
            dt = (time.time() - t0) / 3
        times.append(dt)
        print(f"    L={Lx:>4}: {1000*dt:7.1f} ms")
    # linear-fit quality: time/L should be ~constant for O(N)
    ratios = [times[i] / Ls[i] for i in range(len(Ls))]
    linear = max(ratios) / min(ratios) < 2.5      # per-step cost roughly constant => O(N)

    g1 = sel_seq > 0.8                             # selective SSM solves long-range recall
    g2 = sel_seq > 3 * max(lti_seq, 0.02) and (sel_seq - lti_seq) > 0.3   # selection is the cause
    g3 = linear                                    # confirmed O(N) in sequence length
    allok = g1 and g2 and g3
    print("\nB3-selective gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] selective SSM solves long-range recall (>80% seq)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] selection is the cause (selective seq >> LTI seq)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] forward cost is O(N) in length (per-step ~constant)")
    print(f"\n  >>> {'PASS — selective O(N) SSM gives long-range cognition without attention O(N^2)' if allok else 'CHECK'}")
    print("\nHonest: small model, synthetic selective-copy, single sequential scan (not the")
    print("hardware parallel scan). The point is the SELECTION mechanism + O(N) cost, the core")
    print("of a post-Transformer backbone — not a tuned LM. Fluent language is NOT claimed.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
