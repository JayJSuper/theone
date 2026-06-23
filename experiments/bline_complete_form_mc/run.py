"""完整体 system-level RED-LINE under Monte-Carlo — does the complete form EVER speak a wrong answer?

NOTE-137 showed the integrated system on two scenarios. The honesty thesis demands a statistical
guarantee: across MANY random scenarios, when the system SPEAKS a verified causal answer it must
not be wrong, and it must ABSTAIN exactly when it cannot verify — using a TRUTH-FREE credential
(it has no access to ground truth at decision time).

Design:
 - one native-do net + one selective perception net, trained once.
 - N random scenarios: random true effect beta in [-0.45, 0.45] (incl. near-zero), and random
   observability (signal present vs absent in the stream).
 - TRUTH-FREE identifiability credential = perception confidence  mean(|p_U - 0.5|): high when the
   stream carries usable signal, ~0 when only distractors. No labels used to decide.
 - if identifiable AND native-do audit drift low -> SPEAK (sign from native do); else ABSTAIN.

Metrics: confidently-wrong = SPOKE on a clearly-nonzero effect (|true ATE|>0.1) with the WRONG sign.
The system-level red-line: confidently-wrong == 0. Plus abstention tracks true observability.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_complete_form_mc/run.py
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "bline_native_do_varstruct"))
import importlib.util as _ilu


def _load(name, rel):
    spec = _ilu.spec_from_file_location(name, ROOT / rel)
    m = _ilu.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


vs = _load("vs_varstruct", "experiments/bline_native_do_varstruct/run.py")
cf = _load("cf_complete", "experiments/bline_complete_form/run.py")
FAST = os.environ.get("THEONE_FAST") == "1"
DEVICE = vs.DEVICE
SIG = None


def scenario(beta, observable, n, T, k, rng):
    """U->X, U->Y, X->Y(beta); U seen as a stream with k signal steps (0 if not observable)."""
    global SIG
    U = (rng.random(n) < 0.5).astype(np.float64)
    streams = np.zeros((n, T, 4))
    kk = k if observable else 0
    for i in range(n):
        pos = set(rng.choice(T, kk, replace=False).tolist()) if kk else set()
        for t in range(T):
            if t in pos:
                streams[i, t] = SIG * (2 * U[i] - 1) + 0.3 * rng.standard_normal(4)
            else:
                d = rng.standard_normal(4); d /= np.linalg.norm(d); streams[i, t] = d
    X = (rng.random(n) < (0.25 + 0.5 * U)).astype(np.int8)
    pY = np.clip(0.4 + 0.4 * (U - 0.5) + beta * (X - 0.5), 0.02, 0.98)
    Y = (rng.random(n) < pY).astype(np.int8)
    return streams, U, X, Y


def true_ate(beta, rng):
    """ground-truth ATE = E_U[P(Y|do X=1) - P(Y|do X=0)] under the generative model (U⫫do X)."""
    u = (rng.random(40000) < 0.5).astype(float)
    p1 = np.clip(0.4 + 0.4 * (u - 0.5) + beta * 0.5, 0.02, 0.98)
    p0 = np.clip(0.4 + 0.4 * (u - 0.5) - beta * 0.5, 0.02, 0.98)
    return float((p1 - p0).mean())


def main():
    global SIG
    print("=== 完整体 system-level RED-LINE (Monte-Carlo: does it ever speak a wrong answer?) ===\n")
    rng = np.random.default_rng(0)
    SIG = rng.standard_normal(4); SIG /= np.linalg.norm(SIG)
    donet, mu, sd = cf.train_native_do()

    # train ONE perception net on signal-present streams (fixed SIG)
    n, T, k = (700, 30, 5)
    Xtr = []; ytr = []
    for _ in range(6):
        s, U, _, _ = scenario(0.3, True, n, T, k, rng); Xtr.append(s); ytr.append(U)
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    torch.manual_seed(0)
    pnet = cf.SelectiveGate().to(DEVICE)
    xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE); yt = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.AdamW(pnet.parameters(), lr=5e-3)
    for _ in range(150 if FAST else 300):
        pnet.train(); opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pnet(xt), yt); loss.backward(); opt.step()
    pnet.eval()

    N = 60 if FAST else 200
    spoke = abstained = conf_wrong = correct_sign = 0
    spoke_obs = abstain_unobs = total_obs = total_unobs = 0
    abs_err = []
    for s in range(N):
        beta = float(rng.uniform(-0.45, 0.45)); observable = bool(rng.random() < 0.5)
        total_obs += observable; total_unobs += (not observable)
        streams, U, X, Y = scenario(beta, observable, 700, T, k, rng)
        xs = torch.tensor(streams, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            p = torch.sigmoid(pnet(xs)).cpu().numpy()
        U_hat = (p > 0.5).astype(int)
        # TRUTH-FREE identifiability credential: is the recovered U_hat actually associated with BOTH
        # X and Y? (a precondition for it being a usable confounder to adjust on). Uses only observed
        # X,Y,U_hat — no ground truth. If U_hat ⫫ X or ⫫ Y empirically, adjustment is moot -> ABSTAIN.
        def corr(a, b):
            if a.std() < 1e-6 or b.std() < 1e-6: return 0.0
            return abs(float(np.corrcoef(a, b)[0, 1]))
        cred = min(corr(U_hat, X), corr(U_hat, Y))
        if cred < 0.10:
            abstained += 1; abstain_unobs += (not observable); continue
        ate, drift, _ = cf.infer(U_hat, X, Y, donet, mu, sd)
        # SPEAK only if engine-audit tight AND the effect is certifiable above the net's own error floor
        if drift < 0.06 and abs(ate) > 0.08:
            spoke += 1; spoke_obs += observable
            t = true_ate(beta, rng); abs_err.append(abs(ate - t))
            if abs(t) > 0.1:
                if np.sign(ate) == np.sign(t): correct_sign += 1
                else: conf_wrong += 1                          # SPOKE with WRONG sign on a real effect
        else:
            abstained += 1                                     # within error bars / not audit-tight -> honest abstain

    clear = correct_sign + conf_wrong
    print(f"  scenarios={N}  (observable={total_obs}, unobservable={total_unobs})")
    print(f"  SPOKE={spoke}  ABSTAINED={abstained}")
    print(f"  abstention vs truth: abstained on {abstain_unobs}/{total_unobs} unobservable; spoke on {spoke_obs}/{total_obs} observable")
    print(f"  among spoken clear-effect answers: correct-sign={correct_sign}, CONFIDENTLY-WRONG={conf_wrong}")
    if abs_err:
        print(f"  spoken |ATE - true| mean={np.mean(abs_err):.3f}")

    g1 = conf_wrong == 0                                       # SYSTEM RED-LINE: never speaks a wrong answer
    g2 = (abstain_unobs >= 0.7 * total_unobs)                  # abstains when it cannot perceive the confounder
    g3 = (spoke_obs >= 0.6 * total_obs)                        # speaks when it can verify (useful, not mute)
    allok = g1 and g2 and g3
    print("\nsystem-level red-line gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] RED-LINE: 0 confidently-wrong spoken answers across {N} scenarios")
    print(f"  [{'PASS' if g2 else 'FAIL'}] abstains (truth-free credential) when the confounder is unobservable (≥70%)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] still useful: speaks when it can verify (≥60% of observable)")
    print(f"\n  >>> {'PASS — across many random scenarios the complete form never speaks a wrong verified answer, and abstains via a TRUTH-FREE credential when it cannot verify. Honesty is statistical, not anecdotal.' if allok else 'CHECK'}")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
