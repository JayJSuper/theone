"""B4 extrapolation LIMIT — how far does the internalized do-calculus algorithm reach?

NOTE-139: a message-passing do() net trained on K∈{4,5} extrapolates to K=6,7. The scientific
question: where does it STOP? We train once on K∈{4,5} and sweep the test size K=6..10, reporting
the native-do MAE curve against the exact-enumeration engine and the confounded baseline. A graceful
curve = the algorithm genuinely generalizes; a cliff = it memorized a size-bounded approximation.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_native_do_gnn_limit/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "bline_native_do_gnn"))
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("gnn_mod", ROOT / "experiments/bline_native_do_gnn/run.py")
g = _ilu.module_from_spec(_spec); sys.modules["gnn_mod"] = g; _spec.loader.exec_module(g)
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = g.DEVICE


def main():
    torch.manual_seed(0)
    print("=== B4 extrapolation LIMIT · train K∈{4,5}, sweep test K=6..10 ===\n")
    ntr = 4000 if FAST else 16000
    Gtr = g.make([4, 5], ntr, 0)
    net = g.GNNdo().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-5)
    epochs = 6 if FAST else 18
    for ep in range(epochs):
        net.train(); rng = np.random.default_rng(100 + ep); order = rng.permutation(len(Gtr))
        for c, i in enumerate(order):
            nf, edges, X, Y, d, K, _ = Gtr[i]
            t = torch.tensor(nf, device=DEVICE)
            loss = (net(t, edges, X, Y) - torch.tensor(d, dtype=torch.float32, device=DEVICE)) ** 2
            loss.backward()
            if (c + 1) % 32 == 0:
                opt.step(); opt.zero_grad()
        opt.step(); opt.zero_grad()
    print(f"  trained on K∈{{4,5}} ({ntr} graphs, {epochs} epochs)\n")

    Ks = [6, 7, 8, 9, 10]
    nper = 250 if FAST else 500
    print("   K   native-do MAE   confounded-baseline   ratio(net/base)   verdict")
    curve = {}
    for K in Ks:
        G = g.make([K], nper, 50 + K)
        mae, obs = g.evaluate(net, G)
        ratio = mae / obs if obs > 0 else 9.9
        verdict = "tight" if mae < 0.07 else "loose" if mae < 0.12 else "BREAKS"
        curve[K] = (mae, obs, ratio)
        print(f"   {K:>2}     {mae:.4f}          {obs:.4f}              {ratio:.2f}          {verdict}")

    # the "limit" = largest K where native-do still clearly beats the confounded baseline AND stays <0.10
    reach = max([K for K in Ks if curve[K][0] < 0.10 and curve[K][2] < 0.75], default=0)
    train_max = 5
    print(f"\n  extrapolation reach: trained to K={train_max}, still adjusts correctly out to K={reach} "
          f"(+{reach - train_max} variables beyond training)")
    g1 = reach >= 8                                          # reaches meaningfully beyond K=7 (NOTE-139)
    g2 = all(curve[K][0] <= curve[K - 1][0] + 0.04 for K in Ks[1:])   # degrades gracefully, no cliff
    allok = g1 and g2
    print("\nextrapolation-limit gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] internalized algorithm reaches >= K=8 (beyond NOTE-139's K=7)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] graceful degradation across K=6..10 (no cliff)")
    print(f"\n  >>> {'PASS — the do-calculus algorithm generalizes well past the training size; reach measured, degradation graceful — strong evidence of true algorithm internalization, with an honest measured limit.' if allok else 'CHECK — extrapolation limit found (honest negative is still informative)'}")
    print("\nHonest: amortized from given graph+CPTs; exact enumeration is the recomputable oracle; the")
    print("measured reach IS the result whether it is large or small — we report where it stops.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
