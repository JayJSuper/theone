"""④ B3 selective perception AT SCALE, stitched to native verifiable do.

O(N) SSM complexity is already shown (bline_ssm_scaling). The readiness gap: does the CAUSAL pipeline
hold as the input stream grows long? A latent confounder U is buried as k signal steps among a long
haystack of T-k same-magnitude distractors; a selective (input-gated) SSM must SELECT the signal in
O(N) and recover U; that U_hat then feeds native do(X). We sweep T and check that (a) selective
recovery stays accurate as the haystack grows (selection scales), and (b) the downstream causal
conclusion (ATE sign/size) stays correct — i.e. perception-to-do survives real sequence length.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_b3_perception_scale/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "bline_complete_form"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("cf_mod", ROOT / "experiments/bline_complete_form/run.py")
cf = _ilu.module_from_spec(_spec); sys.modules["cf_mod"] = cf; _spec.loader.exec_module(cf)
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = cf.DEVICE
DIN = 4


def make_stream(n, T, k, rng):
    """U in {0,1}; visible only as length-T stream: k signal steps (encode U) among distractors."""
    U = (rng.random(n) < 0.5).astype(np.float64)
    sig = rng.standard_normal(DIN); sig /= np.linalg.norm(sig)
    S = np.zeros((n, T, DIN), np.float32)
    for i in range(n):
        pos = set(rng.choice(T, k, replace=False).tolist())
        for t in range(T):
            if t in pos:
                S[i, t] = sig * (2 * U[i] - 1) + 0.3 * rng.standard_normal(DIN)
            else:
                d = rng.standard_normal(DIN); d /= np.linalg.norm(d); S[i, t] = d
    return S, U


class SelectiveSSM(nn.Module):
    """selective O(N) gated WEIGHTED-pool: pool = Σ_t g_t⊙v_t / Σ_t g_t, with input-dependent gate g_t
    and value v_t. One linear pass + a reduction = O(N), fully vectorized. Crucially the weighted
    average is SCALE-INVARIANT: the few high-gate signal steps dominate regardless of stream length,
    so selection does not dilute as the haystack grows (unlike a mean/EMA pool)."""
    def __init__(self, d=48):
        super().__init__()
        self.val = nn.Linear(DIN, d)
        # gate is a small MLP: signal steps are ±sig (BOTH U values), so detecting "is a signal step"
        # needs |projection| — a linear gate cannot; a 2-layer gate can. Value stays linear (signed -> U).
        self.gate = nn.Sequential(nn.Linear(DIN, 24), nn.ReLU(), nn.Linear(24, 1))
        self.out = nn.Linear(d, 1)

    def forward(self, x):                                    # x: (B,T,DIN)
        g = torch.sigmoid(self.gate(x))                      # (B,T,1) per-step relevance gate
        v = torch.tanh(self.val(x))                          # (B,T,d)
        pool = (g * v).sum(1) / (g.sum(1) + 1e-6)            # gated weighted average — O(N), no dilution
        return self.out(pool).squeeze(-1)


def perceive_at_T(T, k, n, rng, seed):
    S, U = make_stream(n, T, k, rng)
    X = torch.tensor(S, device=DEVICE); y = torch.tensor(U, dtype=torch.float32, device=DEVICE)
    torch.manual_seed(seed); net = SelectiveSSM().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-3)
    cut = int(0.7 * n)
    for _ in range(300 if FAST else 500):
        net.train(); opt.zero_grad()
        g = torch.sigmoid(net.gate(X[:cut]))                 # sparsity prior: few steps should be relevant
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[:cut]), y[:cut]) + 3e-3 * g.mean()
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Uh = (torch.sigmoid(net(X)) > 0.5).float().cpu().numpy()
    acc = float((Uh[cut:] == U[cut:]).mean())
    return Uh.astype(int), U, acc


def main():
    torch.manual_seed(0)
    print("=== ④ B3 selective perception AT SCALE -> native do (perception survives long streams) ===\n")
    donet, mu, sd = cf.train_native_do()
    rng = np.random.default_rng(0)
    Ts = [128, 512, 2048] if FAST else [256, 1024, 4096, 16384]
    n = 700 if FAST else 1200
    k = 5
    print(f"  signal steps k={k} buried among T-k distractors; selective SSM O(N) scan -> U_hat -> native do\n")
    accs = []; do_ok = []
    for T in Ts:
        Uh, U, acc = perceive_at_T(T, k, n, rng, seed=T)
        # build a confounded SCM from the recovered U, run native do, compare to truth
        Xv = (rng.random(n) < (0.25 + 0.5 * U)).astype(np.int8)
        Yv = (rng.random(n) < np.clip(0.2 + 0.4 * U + 0.3 * Xv, 0, 1)).astype(np.int8)
        ate, drift, _ = cf.infer(Uh, Xv, Yv, donet, mu, sd)
        accs.append(acc); do_ok.append(ate > 0.05 and drift < 0.06)    # true effect is +; must recover positive
        print(f"  T={T:>5}:  haystack ratio {k}/{T}={k/T:.4f}  ·  U-recovery acc={acc:.2f}  ·  native do ATE={ate:+.3f} (audit {drift:.3f})  {'OK' if do_ok[-1] else '—'}")

    g1 = min(accs) > 0.8                                     # selection stays accurate as haystack grows
    g2 = accs[-1] > 0.8                                      # holds at the LONGEST stream
    g3 = all(do_ok)                                          # downstream causal conclusion correct at every T
    allok = g1 and g2 and g3
    print("\nperception-at-scale gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] selective recovery stays accurate across growing haystacks (>0.8)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] holds at the LONGEST stream T={Ts[-1]} (selection scales)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] downstream native-do conclusion stays correct at every length")
    msg = ("PASS — selective O(N) perception recovers the buried confounder even as the stream grows long, "
           "and the native verifiable do() downstream stays correct: perception-to-cognition survives real "
           "sequence length. B3 stitched to B4 at scale.") if allok else "CHECK"
    print(f"\n  >>> {msg}")
    print("\nHonest: O(N) complexity established separately (bline_ssm_scaling); here the focus is that")
    print("SELECTION + the causal conclusion survive long streams. Native do amortized; engine is oracle.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
