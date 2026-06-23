"""Tests for the integrated native verifiable engine (the complete-form heart)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from theone.native import NativeVerifiableEngine


def _gen(n, beta, latent, seed):
    rng = np.random.default_rng(seed)
    u = (rng.random(n) < 0.45).astype(int)
    x = (rng.random(n) < np.where(u == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .25, (0, 1): min(.25 + beta, .95), (1, 0): .65, (1, 1): min(.65 + beta, .95)}
    y = np.array([1 if rng.random() < py[(uu, xx)] else 0 for uu, xx in zip(u, x)])
    cols = {"X": x, "Y": y} if latent else {"U": u, "X": x, "Y": y}
    return pd.DataFrame(cols)


def test_strong_identified_effect_is_trustworthy():
    r = NativeVerifiableEngine().estimate(_gen(4000, 0.30, False, 0), confounder="U")
    assert r.zone == "VERIFIABLE" and r.is_trustworthy()
    assert r.replay_ok and r.identifiable and r.e_value >= 2.0


def test_latent_confounder_is_not_verifiable():
    r = NativeVerifiableEngine().estimate(_gen(4000, 0.05, True, 2), confounder=None)
    assert r.zone != "VERIFIABLE" and not r.is_trustworthy()
    assert r.identifiable is False


def test_replay_verifies_and_catches_tampering():
    eng = NativeVerifiableEngine()
    r = eng.estimate(_gen(4000, 0.30, False, 0), confounder="U")
    assert r.chain.verify()[0] is True
    r.chain.steps[0].recorded = [0.1, 0.9]          # tamper
    assert r.chain.verify()[0] is False


def test_credential_is_complete():
    r = NativeVerifiableEngine().estimate(_gen(4000, 0.30, False, 0), confounder="U")
    c = r.credential
    for k in ("claim", "zone", "e_value", "structural_stability", "replay_ok", "chain_hash"):
        assert k in c
    assert c["chain_steps"] == len(r.chain.steps)


def test_complete_form_observational_and_perceived():
    """The CompleteForm engine runs perceive->identify->verify-do as one object, both paths."""
    from theone.native import CompleteForm
    rng = np.random.default_rng(1); n = 6000
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
    py = {(0, 0): .25, (0, 1): .55, (1, 0): .60, (1, 1): .90}
    Y = np.array([1 if rng.random() < py[(x, u)] else 0 for x, u in zip(X, U)])
    Z = (rng.random(n) < 0.5).astype(int)
    df = pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})
    cf = CompleteForm()
    r = cf.analyze(df, treatment="X", outcome="Y", pre_treatment=["U", "Z"])
    assert r.confounders == ["U"]                       # identifies confounder, drops irrelevant Z
    assert r.credential["replay_ok"] and r.zone in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED", "REJECT")
    # perceived path: confounder seen only as a stream
    base = np.sin(np.linspace(0, 3, 40))[None, :] * (2 * U - 1)[:, None]
    streams = (base + rng.normal(scale=1.0, size=(n, 40))).astype(np.float32)
    r2 = cf.analyze(pd.DataFrame({"X": X, "Y": Y}), streams=streams)
    assert r2.perception is not None and r2.credential["replay_ok"]


def test_complete_form_abstains_when_confounder_hidden():
    """Red-line: with the true confounder UNOBSERVED, the estimate is confounded — the complete
    form must NOT certify it (not identifiable) and must abstain, never act on the biased number."""
    from theone.native import CompleteForm
    rng = np.random.default_rng(1); n = 8000
    U = (rng.random(n) < 0.45).astype(int)
    X = (rng.random(n) < np.where(U == 1, 0.85, 0.15)).astype(int)   # strong confounding
    pY = {(0, 0): .30, (0, 1): .70, (1, 0): .35, (1, 1): .75}        # true X effect ~0.05
    Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
    df = pd.DataFrame({"U": U, "X": X, "Y": Y})
    cf = CompleteForm()
    r = cf.analyze(df.drop(columns=["U"]), pre_treatment=[])         # U hidden
    assert r.zone != "VERIFIABLE"                  # cannot certify an unidentifiable estimate
    assert cf.recommend(r)["action"] == "abstain"  # never acts on the confounded number


def test_complete_form_never_acts_on_null_effect():
    """Integrity red-line: across several null-effect regimes the complete form must NEVER
    recommend an action (no false certification)."""
    from theone.native import CompleteForm
    cf = CompleteForm()
    for seed in range(6):
        rng = np.random.default_rng(200 + seed); n = 7000
        U = (rng.random(n) < 0.45).astype(int)
        X = (rng.random(n) < np.where(U == 1, 0.75, 0.25)).astype(int)
        pY = {(0, 0): .35, (0, 1): .62, (1, 0): .35, (1, 1): .62}   # X has ZERO effect
        Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
        Z = (rng.random(n) < 0.5).astype(int)
        df = pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})
        act = cf.recommend(cf.analyze(df, pre_treatment=["U", "Z"]))
        assert act["action"] == "abstain"          # never acts on a true-null effect


def test_complete_form_act_and_abstain():
    """The 'act' step: recommend on a trustworthy effect, abstain on a fragile one."""
    from theone.native import CompleteForm

    def mk(x_effect, seed):
        rng = np.random.default_rng(seed); n = 8000
        U = (rng.random(n) < 0.45).astype(int)
        X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
        e = x_effect
        pY = {(0, 0): .40, (0, 1): .62, (1, 0): min(.40 + e, .97), (1, 1): min(.62 + e, .97)}
        Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
        Z = (rng.random(n) < 0.5).astype(int)
        return pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})

    cf = CompleteForm()
    strong = cf.recommend(cf.analyze(mk(0.30, 1), pre_treatment=["U", "Z"]))
    fragile = cf.recommend(cf.analyze(mk(0.015, 3), pre_treatment=["U", "Z"]))
    assert strong["action"] == "recommend"
    assert fragile["action"] == "abstain"


def test_complete_form_cognitive_loop():
    """CompleteForm conclusions are storable in sovereign memory and recalled by signature."""
    from theone.native import CompleteForm
    from theone.memory.sovereign import SovereignMemory
    from theone.memory.signature import CausalSignature

    def mk(x_effect, seed):
        rng = np.random.default_rng(seed); n = 6000
        U = (rng.random(n) < 0.45).astype(int)
        X = (rng.random(n) < np.where(U == 1, 0.8, 0.2)).astype(int)
        e = x_effect
        pY = {(0, 0): .40, (0, 1): .62, (1, 0): min(.40 + e, .97), (1, 1): min(.62 + e, .97)}
        Y = np.array([1 if rng.random() < pY[(x, u)] else 0 for x, u in zip(X, U)])
        Z = (rng.random(n) < 0.5).astype(int)
        return pd.DataFrame({"U": U, "X": X, "Y": Y, "Z": Z})

    cf = CompleteForm(); mem = SovereignMemory(":memory:")
    rA = cf.analyze(mk(0.30, 1), pre_treatment=["U", "Z"])
    rB = cf.analyze(mk(0.08, 2), pre_treatment=["U", "Z"])
    idA = mem.remember("cohort A", rA.credential, source="cf")
    mem.remember("cohort B", rB.credential, source="cf")
    q = CausalSignature(treatment="X", target="Y", adjustment_set=("U",),
                        effect=rA.effect, regime=rA.credential["regime"])
    top = mem.recall_for_decision(q, k=1)
    assert top[0].mem_id == idA                    # signature recall returns the matching cohort
    assert abs(top[0].signature.effect - rA.effect) < 0.05


def test_complete_form_continuous_cognitive_loop():
    """Continuous conclusions share the cognitive-OS loop: remembered + recalled by signature."""
    from theone.native import CompleteForm
    from theone.memory.sovereign import SovereignMemory
    from theone.memory.signature import CausalSignature
    rng = np.random.default_rng(1); n = 1600
    X = rng.normal(size=(n, 6)).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * X[:, 0] - 0.5 * X[:, 1])))).astype(np.float32)
    y = (2.0 + X[:, 0] + 0.5 * X[:, 2] + 3.0 * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    cf = CompleteForm(); mem = SovereignMemory(":memory:")
    r = cf.analyze_continuous(X, t, y)
    mid = mem.remember("continuous strong", r.credential, source="cf")
    q = CausalSignature(treatment="T", target="Y_continuous",
                        adjustment_set=("<all covariates>",), effect=r.effect,
                        regime=r.credential["regime"])
    assert mem.recall_for_decision(q, k=1)[0].mem_id == mid
    assert cf.recommend(r, min_effect=0.2)["action"] == "recommend"


def test_complete_form_continuous():
    """CompleteForm handles continuous outcomes through the same one-call loop."""
    from theone.native import CompleteForm
    rng = np.random.default_rng(0); n = 1500
    X = rng.normal(size=(n, 6)).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-(0.6 * X[:, 0] - 0.5 * X[:, 1])))).astype(np.float32)
    y = (2.0 + X[:, 0] + 0.5 * X[:, 2] + 3.0 * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    r = CompleteForm().analyze_continuous(X, t, y)
    assert abs(r.effect - 3.0) < 0.6 and r.credential["replay_ok"]
    assert "reproducibility_stability" in r.credential


def _gen_continuous(n, ate, seed):
    """Continuous-outcome SCM: confounder shifts both treatment and a continuous Y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    logit = 0.6 * X[:, 0] - 0.5 * X[:, 1]
    t = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.float32)
    base = 2.0 + X[:, 0] + 0.5 * X[:, 2] - 0.3 * X[:, 3]
    y = base + ate * t + rng.normal(scale=0.5, size=n).astype(np.float32)
    return X, t, y.astype(np.float32)


def test_continuous_outcome_recovers_ate_and_replays():
    X, t, y = _gen_continuous(1500, 3.0, 0)
    r = NativeVerifiableEngine().estimate_continuous(X, t, y, covariate_sufficient=True)
    assert abs(r.effect - 3.0) < 0.6          # recovers the continuous ATE
    assert r.replay_ok                        # reproducible-inference replay holds
    assert r.zone in ("VERIFIABLE", "UNCERTAINTY_QUANTIFIED")
    assert "reproducibility_stability" in r.credential


def test_continuous_unmeasured_confounder_not_verifiable():
    X, t, y = _gen_continuous(1500, 3.0, 1)
    r = NativeVerifiableEngine().estimate_continuous(X, t, y, covariate_sufficient=False)
    assert r.zone != "VERIFIABLE"             # no covariate sufficiency -> never 'verifiable'
    assert r.identifiable is False


def test_continuous_zero_effect_never_certified():
    """Red-line: a genuinely near-zero continuous effect must NEVER be stamped VERIFIABLE —
    the three-zone classifier abstains (REJECT/UNCERTAINTY) when the effect is fragile."""
    X, t, y = _gen_continuous(1500, 0.0, 0)            # true ATE = 0
    r = NativeVerifiableEngine().estimate_continuous(X, t, y, covariate_sufficient=True)
    assert r.zone != "VERIFIABLE"
    assert r.e_value < 2.0                             # E-value reflects fragility


def test_perceive_features_feed_continuous_engine_beats_naive():
    """Perception (SSM features from streams) -> continuous native estimate recovers the ATE
    better than the naive confounded contrast."""
    from theone.native.perception import SSMPerception
    n, T, ate = 1400, 48, 3.0
    rng = np.random.default_rng(0)
    c = rng.normal(size=n).astype(np.float32)
    streams = (np.sin(np.linspace(0, 3, T))[None, :] * c[:, None]
               + rng.normal(scale=1.0, size=(n, T))).astype(np.float32)
    t = (rng.random(n) < 1 / (1 + np.exp(-1.2 * c))).astype(np.float32)
    y = (2.0 + 1.5 * c + ate * t + rng.normal(scale=0.5, size=n)).astype(np.float32)
    naive_err = abs(float(y[t == 1].mean() - y[t == 0].mean()) - ate)
    X = SSMPerception(hidden_dim=24, seed=0).perceive_features(streams, k=4)
    r = NativeVerifiableEngine().estimate_continuous(X, t, y, covariate_sufficient=True)
    assert abs(r.effect - ate) < naive_err    # perceiving+adjusting beats the confounded contrast
    assert r.replay_ok
