"""The integrated native verifiable engine, end-to-end — the complete-form heart.

One call -> a causal estimate that carries a replayable derivation chain (self-verifies,
no oracle) AND a three-zone honest status (can't let a latent-confounded effect pass as
verifiable). Demonstrates: trustworthy answer on identified data; honest downgrade when
the confounder is latent; tamper-evident replay.

Run:  .venv/bin/python experiments/native_engine_demo/run.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.native import NativeVerifiableEngine


def gen(n, beta, latent, seed):
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.45).astype(int)
    x = (rng.random(n) < np.where(u == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .25, (0, 1): min(.25 + beta, .95), (1, 0): .65, (1, 1): min(.65 + beta, .95)}
    y = np.array([1 if rng.random() < py[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    cols = {"X": x, "Y": y} if latent else {"U": u, "X": x, "Y": y}
    return pd.DataFrame(cols)


def main():
    print("=== native verifiable engine · end-to-end (estimate + replay chain + three-zone) ===\n")
    eng = NativeVerifiableEngine()
    ok = True

    cases = [
        ("A 强效应,混杂可观测", gen(4000, 0.30, False, 0), None),
        ("B 弱效应,混杂可观测", gen(4000, 0.05, False, 1), None),
        ("C 混杂潜在(只见X,Y)", gen(4000, 0.05, True, 2), None),
    ]
    for name, df, _ in cases:
        conf = "U" if "U" in df.columns else None
        r = eng.estimate(df, confounder=conf)
        print(f"{name}")
        print(f"   ATE={r.effect}  →  zone={r.zone}  (trustworthy={r.is_trustworthy()})")
        print(f"   E-value={r.e_value} · stability={r.structural_stability} · "
              f"identifiable={r.identifiable} · replay_ok={r.replay_ok}")
        print(f"   chain: {r.chain.root_hash()[:12]}… ({len(r.chain.steps)} pure steps)\n")

    rA = eng.estimate(cases[0][1], confounder="U")
    rC = eng.estimate(cases[2][1], confounder=None)

    # 1. identified strong effect -> trustworthy (verifiable + replay ok)
    a_trust = rA.is_trustworthy()
    # 2. latent confounder -> NOT verifiable (honest downgrade), but still self-consistent
    c_downgraded = rC.zone != "VERIFIABLE"
    # 3. tamper the chain -> replay catches it
    rT = eng.estimate(cases[0][1], confounder="U")
    rT.chain.steps[0].recorded = [0.1, 0.9]
    tamper_caught = not rT.chain.verify()[0]

    ok = a_trust and c_downgraded and tamper_caught
    print("native-engine gate:")
    print(f"  identified strong effect -> trustworthy ........... {'PASS' if a_trust else 'FAIL'}")
    print(f"  latent confounder -> honestly NOT verifiable ...... {'PASS' if c_downgraded else 'FAIL'}")
    print(f"  tampered derivation chain -> caught by replay ..... {'PASS' if tamper_caught else 'FAIL'}")
    print(f"\n  >>> {'PASS — one native conclusion: replay-verified + three-zone status, integrated' if ok else 'CHECK'}")
    print("\nThis is the complete-form heart: the de-risked B-line pieces (B1 learned adjustment,")
    print("Q1 replay chain, Q3 three-zone, E-value) composed into ONE first-class engine in")
    print("src/theone/native/ — a learned causal estimate that self-verifies and self-classifies.")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
