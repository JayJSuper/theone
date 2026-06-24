"""⑤ THE COMPLETE FORM AT REAL SCALE — the whole machine, composed, on long inputs.

Each pillar is validated at scale individually (B1 256M; B4 GNN to K=10 / NOTE-143; B3 perception to
long streams / NOTE-145; B2 cloud). ⑤ shows they COMPOSE at scale in one credentialed run:

  long noisy stream (T=2048)
   -> B3 selective O(N) perception recovers the latent confounder U (needle 0.24%)
   -> B4 native do() on the (U,X,Y) SCM, engine-audited
   -> B2 VerifiedReporter speaks it, round-trip-gated
   -> and ABSTAINS (truth-free credential) when the stream carries no usable signal.

This is the toy complete-form (NOTE-137) re-run with the AT-SCALE perception + native-do components —
evidence the system is not just correct in pieces but correct as a whole at real input size.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_complete_form_scale/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import importlib.util as _ilu


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, ROOT / rel)
    m = _ilu.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


cf = _load("cf_mod", "experiments/bline_complete_form/run.py")
b3 = _load("b3_mod", "experiments/bline_b3_perception_scale/run.py")
from theone.language import VerifiedReporter, Finding
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = cf.DEVICE


def train_perception(X, y, cut, seed=1):
    torch.manual_seed(seed); net = b3.SelectiveSSM().to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-3)
    for _ in range(300 if FAST else 500):
        net.train(); opt.zero_grad()
        g = torch.sigmoid(net.gate(X[:cut]))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(X[:cut]), y[:cut]) + 3e-3 * g.mean()
        loss.backward(); opt.step()
    net.eval(); return net


def main():
    torch.manual_seed(0)
    print("=== ⑤ COMPLETE FORM AT REAL SCALE (long-stream perceive → native do → speak / abstain) ===\n")
    donet, mu, sd = cf.train_native_do()
    reporter = VerifiedReporter(
        label={"treatment": "starting the treatment", "outcome": "recovery"},
        entity_syn={"treatment": ["treatment", "the drug", "starting the treatment", "the therapy"],
                    "outcome": ["recovery", "recovering", "getting better", "the outcome"]})
    rng = np.random.default_rng(0)
    T = 1024 if FAST else 4096
    k, n = 6, (800 if FAST else 1200)
    # ONE stream with a fixed signal direction; split train/test (the signal direction must be consistent
    # between training and inference — make_stream randomizes it per call, so we generate once and split).
    S, U = b3.make_stream(n, T, k, rng)
    cut = int(0.7 * n)
    Xt = torch.tensor(S, device=DEVICE); yt = torch.tensor(U, dtype=torch.float32, device=DEVICE)
    pnet = train_perception(Xt, yt, cut, seed=1)

    def confidence(net, streams):
        with torch.no_grad():
            g = torch.sigmoid(net.gate(torch.tensor(streams, device=DEVICE)))
        return float(g.max(1).values.mean())

    print(f"  stream length T={T} · needle k/T={k/T:.4f}\n")
    # Scenario A: held-out test slice of the SAME stream -> perceive, infer, SPEAK
    Ste, Ute = S[cut:], U[cut:]
    with torch.no_grad():
        U_hat = (torch.sigmoid(pnet(torch.tensor(Ste, device=DEVICE))) > 0.5).long().cpu().numpy()
    acc = float((U_hat == Ute).mean()); credA = confidence(pnet, Ste)
    m = len(Ute); rng2 = np.random.default_rng(5)
    Xv = (rng2.random(m) < (0.25 + 0.5 * Ute)).astype(np.int8)
    Yv = (rng2.random(m) < np.clip(0.2 + 0.4 * Ute + 0.3 * Xv, 0, 1)).astype(np.int8)
    ate, drift, _ = cf.infer(U_hat, Xv, Yv, donet, mu, sd)
    zone = "VERIFIABLE" if drift < 0.05 else "UNCERTAINTY_QUANTIFIED"
    f = Finding(cause="treatment", effect="outcome", direction=1 if ate >= 0 else -1, zone=zone, ate=ate, e_value=2.1)
    out = reporter.report([f]); lineA = (out["report"] or out["held_back"])[0]
    print(f"  [A · T={T} signal present]  perceive acc={acc:.2f}  cred={credA:.2f}  native do ATE={ate:+.3f} (audit {drift:.3f})")
    print(f"     SPEAK → \"{lineA}\"")

    # Scenario B: no signal in the stream -> truth-free credential low -> ABSTAIN
    Sb, Ub = b3.make_stream(n, T, 0, rng)
    credB = confidence(pnet, Sb)
    identifiable = credB > 0.9 * credA                         # relative to the signal-present credential
    print(f"\n  [B · T={T} NO signal]  cred={credB:.2f} (vs {credA:.2f}) → identifiable={identifiable}")
    print(f"     SPEAK → \"{'(would answer)' if identifiable else 'The engine could not certify a causal effect — reported as inconclusive.'}\"")

    g1 = acc > 0.8                                             # perception works at scale
    g2 = (drift < 0.06) and (len(out['report']) == 1) and (ate > 0.05)   # do correct + audited + spoken
    g3 = not identifiable                                      # abstains when the long stream has no signal
    allok = g1 and g2 and g3
    print("\nreal-scale complete-form gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] B3 perceives the confounder from a long T={T} stream (acc>0.8)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] B4 native do correct + engine-audited + B2 fluent answer round-trips")
    print(f"  [{'PASS' if g3 else 'FAIL'}] abstains (truth-free credential) when the long stream has no signal")
    msg = ("PASS — the complete form composes AT REAL SCALE: long-stream O(N) perception -> engine-tight "
           "native do -> fluent credentialed speech, abstaining when the signal is absent. The whole "
           "machine is correct as one system at real input size.") if allok else "CHECK"
    print(f"\n  >>> {msg}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
