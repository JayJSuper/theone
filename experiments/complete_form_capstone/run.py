"""Complete-form capstone — ONE engine runs perceive -> identify -> verify-do -> credential.

Two end-to-end paths through the single CompleteForm object:
  A. OBSERVATIONAL: raw data with pre-treatment candidates {U(confounder), Z(irrelevant)} and a
     post-treatment collider C. The engine IDENTIFIES the confounder from data and does a
     replay-verified do() on it — matching truth, beating naive choices.
  B. PERCEIVED: the confounder is NOT a column — it is observed only as a noisy STREAM. The
     engine PERCEIVES it (SSM front-end) and does verified do() on the recovered confounder.

One object, one credential per call — the de-risked native pieces composed into the complete form.

Run:  .venv/bin/python experiments/complete_form_capstone/run.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from theone.native import CompleteForm


def make_obs(n, seed):
    rng = np.random.default_rng(seed)
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    pY = {(0, 0): .25, (0, 1): .55, (1, 0): .60, (1, 1): .90}
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    C = ((X + Y + (rng.random(n) < 0.3)) >= 2).astype(int)
    Z = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"U": U, "X": X, "Y": Y, "C": C, "Z": Z})


def make_perceived(n, T, seed):
    rng = np.random.default_rng(seed)
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    pY = {(0, 0): .25, (0, 1): .55, (1, 0): .60, (1, 1): .90}
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    # U observed only as a noisy stream
    base = np.sin(np.linspace(0, 3, T))[None, :] * (2 * U - 1)[:, None]
    streams = (base + rng.normal(scale=1.0, size=(n, T))).astype(np.float32)
    return pd.DataFrame({"X": X, "Y": Y}), streams


def true_ate(seed=0, n=400000):
    rng = np.random.default_rng(seed)
    U = (rng.random(n) < 0.45).astype(int)
    pY = {(0, 0): .25, (0, 1): .55, (1, 0): .60, (1, 1): .90}
    y1 = np.array([1 if rng.random() < pY[(1, u)] else 0 for u in U]).mean()
    y0 = np.array([1 if rng.random() < pY[(0, u)] else 0 for u in U]).mean()
    return float(y1 - y0)


def main():
    print("=== complete-form capstone · ONE engine: perceive -> identify -> verify-do ===\n")
    cf = CompleteForm()
    truth = true_ate()
    print(f"true do(X=1)-do(X=0) = {truth:+.3f}\n")

    # Path A — observational: identify the confounder from data
    dfA = make_obs(8000, seed=1)
    rA = cf.analyze(dfA, treatment="X", outcome="Y", pre_treatment=["U", "Z"])
    print("PATH A · observational")
    print(f"  identified confounders : {rA.confounders}")
    print(f"  do = {rA.effect:+.3f}  |bias| {abs(rA.effect-truth):.3f}  zone={rA.zone}  replay={rA.credential['replay_ok']}")

    # Path B — perceived: confounder seen only as a stream
    dfB, streams = make_perceived(3000, 48, seed=2)
    rB = cf.analyze(dfB, treatment="X", outcome="Y", streams=streams)
    print("\nPATH B · perceived (confounder from stream)")
    print(f"  perception : {rB.perception}")
    print(f"  do = {rB.effect:+.3f}  |bias| {abs(rB.effect-truth):.3f}  zone={rB.zone}  replay={rB.credential['replay_ok']}")

    gA = abs(rA.effect - truth) < 0.05 and rA.confounders == ["U"] and rA.credential["replay_ok"]
    gB = abs(rB.effect - truth) < 0.12 and rB.credential["replay_ok"]    # perception adds noise
    gC = rA.trustworthy                                                  # observational path certified
    allok = gA and gB and gC
    print("\ncapstone gate:")
    print(f"  [{'PASS' if gA else 'FAIL'}] observational: identify {{U}} + verified do matches truth")
    print(f"  [{'PASS' if gB else 'FAIL'}] perceived: recover confounder from stream + verified do")
    print(f"  [{'PASS' if gC else 'FAIL'}] observational conclusion is trustworthy (zone+replay)")
    print(f"\n  >>> {'PASS — the complete form runs as ONE engine, end to end, credentialed' if allok else 'CHECK'}")
    print("\nHonest: toy SCMs, the regimes each piece was de-risked in. Fluent language is NOT in")
    print("this loop (B2 open). The point is the COMPOSITION: one object, one credential per call,")
    print("perception provenance + identification rationale + replay-verified do — the complete form.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
