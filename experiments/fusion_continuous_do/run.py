"""Fusion deepening · the L1->L2 connector: verifiable de-confounded do() on a
continuous latent, as a CredentialedLayer (promoting native_causal_latent probes 4/5).

L1 yields a continuous latent / proxy stream for a latent confounder; this layer
estimates P(Y=1|do(X=1)) through a learned backdoor adjustment and gates on the
truth-free self-checks: subset-spread bias indicator + split-half recompute. Clean
proxies ANSWER (do near truth, declared residual); noisy/sparse proxies ABSTAIN.

Run:  .venv/bin/python experiments/fusion_continuous_do/run.py
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from theone.layer2_world_model import ContinuousCausalLayer

A, B, C = 1.0, 1.5, 1.8


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -700, 700)))


def truth_do_x1():
    v, _ = quad(lambda u: sig(B + C * u) * norm.pdf(u), -np.inf, np.inf)
    return float(v)


def gen(p, sigma, N, seed):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(N)                          # latent confounder (unobserved)
    alphas = rng.uniform(0.6, 1.4, p) * rng.choice([-1.0, 1.0], p)
    P = u[:, None] * alphas[None, :] + rng.normal(0, sigma, (N, p))
    x = (rng.random(N) < sig(A * u)).astype(float)
    y = (rng.random(N) < sig(B * x + C * u)).astype(float)
    return P, x, y


def main():
    print("=== Fusion deepening: L1->L2 connector (verifiable continuous-latent do) ===\n")
    truth = truth_do_x1()
    print(f"(evaluation-only) ground truth do(X=1) = {truth:.4f}\n")
    L = ContinuousCausalLayer()
    ok = True

    cases = [("clean  p=8 σ=0.4", 8, 0.4, 30000),
             ("ok     p=8 σ=0.8", 8, 0.8, 30000),
             ("noisy  p=8 σ=1.6", 8, 1.6, 30000),
             ("sparse p=1 σ=0.8", 1, 0.8, 30000)]
    decisions = {}
    for label, p, sigma, N in cases:
        P, x, y = gen(p, sigma, N, seed=11 + p + int(10 * sigma))
        v = L.run({"proxies": P, "treatment": x, "outcome": y})
        if v.is_answer():
            _, info = v.credential.verify()
            resid = abs(v.credential.value - truth)
            print(f"  {label:<18} ANSWER  do={v.credential.value:.4f} "
                  f"(true resid {resid:.4f}) | spread={v.credential.evidence['subset_spread']:.4f} "
                  f"| recompute gap={info.get('gap', 0):.1e}")
            decisions[label] = ("ANSWER", resid)
        else:
            print(f"  {label:<18} ABSTAIN: {v.reason}")
            decisions[label] = ("ABSTAIN", None)

    # contract: clean/ok ANSWER with small residual; noisy/sparse ABSTAIN
    ok &= decisions["clean  p=8 σ=0.4"][0] == "ANSWER" and decisions["clean  p=8 σ=0.4"][1] < 0.02
    ok &= decisions["ok     p=8 σ=0.8"][0] == "ANSWER"
    ok &= decisions["noisy  p=8 σ=1.6"][0] == "ABSTAIN"
    ok &= decisions["sparse p=1 σ=0.8"][0] == "ABSTAIN"

    print("\nL1->L2 connector contract: continuous latent -> de-confounded do() with a")
    print("DECLARED, recomputable residual; the layer ANSWERs only when the truth-free bias")
    print("indicator certifies low proxy-incompleteness bias, else ABSTAINS. This is the")
    print("native_causal_latent probe line, now a spine layer between perception and causal.")
    print(f"\nself-test: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
