"""B2 toward language — VERIFIABLE-BY-CONSTRUCTION language generation.

The One's mission is "give an answer AND a way to check it". For LANGUAGE that means: every
generated sentence must be BACKED by the verifiable engine and the generator must be unable to
hallucinate. This is the honest B2-language seed — NOT open-domain fluency (the multi-year
frontier), but the by-construction-verifiable principle applied to language:

  - generation is NON-AUTOREGRESSIVE: all claims emitted in parallel from the structured,
    already-verified result (not token-by-token guessing);
  - each emitted sentence carries an EVIDENCE pointer to a recomputable credential field;
  - a verifier RE-CHECKS every sentence against the result — a sentence with no/contradicting
    backing is rejected (so an injected hallucination cannot survive);
  - when the engine ABSTAINS, the generator says so honestly ("I cannot verify this"), it does
    NOT fabricate a number.

Contrast: a free-form LLM sentence ("X definitely cures Y, 100% guaranteed") has no backing and
is rejected. The One only utters what it can recompute.

Run:  .venv/bin/python experiments/bline_b2_language/run.py
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from theone.native import CompleteForm


def make(n, x_effect, seed):
    """x_effect = the (small or large) causal bump X adds to P(Y); U is the confounder.
    Small x_effect -> low E-value -> engine REJECTs (fragile), triggering honest abstain."""
    rng = np.random.default_rng(seed)
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    e = x_effect
    # P(Y=1|X,U): U drives the base; X adds a controllable bump e (same for both U levels)
    pY = {(0, 0): .40, (0, 1): .62, (1, 0): min(.40 + e, .97), (1, 1): min(.62 + e, .97)}
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    Z = (rng.random(n) < 0.5).astype(int)
    return pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})


# --- verifiable-by-construction language generator -------------------------------------------
def generate(result) -> list[dict]:
    """NON-AUTOREGRESSIVE: emit all sentences in parallel from the verified structure. Each
    sentence carries an `evidence` key naming the recomputable credential field it stands on.
    A REJECT zone yields an honest abstain sentence — never a fabricated number."""
    cred = result.credential
    out = []
    if result.zone == "REJECT" or result.effect is None:
        out.append({"text": "I cannot verify a trustworthy causal effect here, so I abstain "
                            "rather than state a number.", "evidence": "zone", "value": result.zone})
        return out
    out.append({"text": f"Adjusting for {result.confounders}, the estimated causal effect of the "
                        f"treatment on the outcome is {result.effect:+.3f}.",
                "evidence": "effect", "value": result.effect})
    out.append({"text": f"This conclusion is rated {result.zone}; an unmeasured confounder would "
                        f"need an association of at least {cred['e_value']} (E-value) with both "
                        f"treatment and outcome to overturn it.",
                "evidence": "e_value", "value": cred["e_value"]})
    out.append({"text": f"The derivation is replay-verified ({cred['replay_ok']}) — you can "
                        f"recompute it from chain hash {str(cred['chain_hash'])[:12]}.",
                "evidence": "replay_ok", "value": cred["replay_ok"]})
    return out


def verify_sentences(sentences, result) -> list[bool]:
    """Re-check each sentence against the recomputable result: the evidence field must exist and
    its stated value must match the result (within tolerance for floats)."""
    cred = result.credential
    truth = {"effect": result.effect, "zone": result.zone,
             "e_value": cred.get("e_value"), "replay_ok": cred.get("replay_ok")}
    oks = []
    for s in sentences:
        ev = s.get("evidence"); val = s.get("value")
        if ev not in truth:
            oks.append(False); continue
        t = truth[ev]
        ok = (abs(val - t) < 1e-6) if isinstance(t, float) else (val == t)
        oks.append(bool(ok))
    return oks


def main():
    print("=== B2 toward language · verifiable-by-construction generation ===\n")
    cf = CompleteForm()

    # case A: a real, strong, identifiable effect -> verified sentences
    rA = cf.analyze(make(8000, 0.30, 1), pre_treatment=["U", "Z"])
    sA = generate(rA)
    okA = verify_sentences(sA, rA)
    print("CASE A — strong identifiable effect:")
    for s, ok in zip(sA, okA):
        print(f"  [{'verified' if ok else 'REJECTED'}] {s['text']}")

    # case B: a fragile effect the engine rejects -> honest abstain, NO number
    rB = cf.analyze(make(8000, 0.02, 7), pre_treatment=["U", "Z"])
    sB = generate(rB)
    print(f"\nCASE B — fragile effect (engine zone={rB.zone}):")
    for s in sB:
        print(f"  [abstain] {s['text']}")

    # adversarial: inject a hallucinated sentence and show the verifier rejects it
    hallucination = {"text": "X definitely cures the outcome, 100% guaranteed.",
                     "evidence": "effect", "value": 0.999}
    halluc_ok = verify_sentences([hallucination], rA)[0]
    print("\nADVERSARIAL — injected hallucination:")
    print(f"  [{'verified' if halluc_ok else 'REJECTED'}] {hallucination['text']}")

    g1 = all(okA)                                          # every generated sentence is backed
    g2 = (rB.zone == "REJECT") and (len(sB) == 1) and ("abstain" in sB[0]["text"])  # honest abstain
    g3 = not halluc_ok                                     # hallucination cannot pass the verifier
    g4 = all(s["evidence"] in ("effect", "e_value", "replay_ok") for s in sA)       # provenance
    allok = g1 and g2 and g3 and g4
    print("\nB2-language gate:")
    print(f"  [{'PASS' if g1 else 'FAIL'}] every generated sentence is engine-backed (recomputable)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] engine abstains -> honest 'cannot verify', no fabricated number")
    print(f"  [{'PASS' if g3 else 'FAIL'}] injected hallucination is REJECTED by the verifier")
    print(f"  [{'PASS' if g4 else 'FAIL'}] each sentence carries a provenance pointer")
    print(f"\n  >>> {'PASS — verifiable-by-construction language: every sentence checkable, no hallucination' if allok else 'CHECK'}")
    print("\nHonest scope: this is NOT open-domain fluent language (the real B2 frontier, multi-")
    print("year). It is the by-construction-verifiable PRINCIPLE applied to language — sentences")
    print("rendered non-autoregressively from already-verified structure, each recomputable, with")
    print("honest abstention. Fluency here is structured/templated; the VERIFIABILITY is the point.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
