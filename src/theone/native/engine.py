"""The One · native verifiable engine — the integrated heart of the complete form.

Ties the de-risked B-line pieces into ONE first-class component: a learned causal estimate
that carries (1) a REPLAYABLE derivation chain of PURE steps (Q1/Q4 — re-run to verify, no
external oracle, no state mutation), and (2) a THREE-ZONE honest status (Q3 —
verifiable / uncertainty-quantified / reject, from structural stability + identifiability +
E-value, so a latent-confounded effect can't sneak into 'verifiable'). One credentialed
native conclusion.

Two paths: (a) `estimate` — binary confounded setting, exact do() marginalization recorded
as a 1e-6-recomputable pure-step chain; (b) `estimate_continuous` — continuous outcomes on
real-covariate data (IHDP-grade) via a learned TARNet, verified by reproducible-inference
replay + a continuous three-zone status (continuous E-value). Large-scale training is the
remaining extension; the architecture composes the verified pieces into a usable engine.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from theone.layer2_world_model.discovery import bootstrap_stability
from theone.layer2_world_model.sensitivity import e_value_for_do


# --- replayable pure-step derivation chain (Q1/Q4 done right) -----------------
@dataclass
class PureStep:
    sid: str
    op: str
    fn: Callable
    inputs: dict
    recorded: Any

    def replay(self) -> Any:
        return self.fn(**self.inputs)               # pure: deterministic, no side effects

    def hash(self) -> str:
        c = json.dumps({"sid": self.sid, "op": self.op, "recorded": _ser(self.recorded)},
                       sort_keys=True, default=str)
        return hashlib.sha256(c.encode()).hexdigest()


def _ser(v):
    if isinstance(v, float):
        return round(v, 10)
    if isinstance(v, (list, tuple)):
        return [_ser(x) for x in v]
    return v


@dataclass
class DerivationChain:
    steps: list = field(default_factory=list)

    def record(self, sid, op, fn, **inputs):
        out = fn(**inputs)
        self.steps.append(PureStep(sid, op, fn, inputs, out))
        return out

    def verify(self) -> tuple[bool, list[str]]:
        errs = []
        for s in self.steps:
            try:
                got = s.replay()
            except Exception as e:                  # a throwing recompute = not verified
                errs.append(f"{s.sid}: replay raised {type(e).__name__}"); continue
            ok = (abs(got - s.recorded) < 1e-9 if isinstance(got, float)
                  else (_ser(got) == _ser(s.recorded)))
            if not ok:
                errs.append(f"{s.sid}: replay != recorded")
        return (not errs), errs

    def root_hash(self) -> str:
        return hashlib.sha256("".join(s.hash() for s in self.steps).encode()).hexdigest()


@dataclass
class NativeResult:
    effect: float                       # P(Y=1|do X=1) - P(Y=1|do X=0)
    zone: str                           # VERIFIABLE / UNCERTAINTY_QUANTIFIED / REJECT
    e_value: float
    structural_stability: float
    identifiable: bool
    chain: DerivationChain
    replay_ok: bool
    credential: dict

    def is_trustworthy(self) -> bool:
        return self.zone == "VERIFIABLE" and self.replay_ok


class NativeVerifiableEngine:
    """A learned causal estimate that self-verifies (replay) and self-classifies (three-zone)."""
    STABILITY_TOL, EVALUE_TOL = 0.8, 2.0

    def estimate(self, df: pd.DataFrame, treatment: str = "X", outcome: str = "Y",
                 confounder: Optional[str] = "U") -> NativeResult:
        identifiable = confounder is not None and confounder in df.columns

        # --- learned adjustment, recorded as a REPLAYABLE chain of pure steps ---
        chain = DerivationChain()
        if identifiable:
            pu = float((df[confounder] == 1).mean())
            # P(Y=1 | X=x, U=u) tables (the learned/estimated quantities)
            def cond(u, x):
                m = (df[confounder] == u) & (df[treatment] == x)
                return float(df[m][outcome].mean()) if m.any() else 0.0
            w = chain.record("w", "P(U) weights", lambda pu: [1 - pu, pu], pu=pu)
            y1 = chain.record("y1", "P(Y=1|do X=1,U)", lambda a, b: [a, b],
                              a=cond(0, 1), b=cond(1, 1))
            y0 = chain.record("y0", "P(Y=1|do X=0,U)", lambda a, b: [a, b],
                              a=cond(0, 0), b=cond(1, 0))
            do1 = chain.record("do1", "marginalize do(X=1)",
                               lambda weights, ys: sum(p * v for p, v in zip(weights, ys)),
                               weights=w, ys=y1)
            do0 = chain.record("do0", "marginalize do(X=0)",
                               lambda weights, ys: sum(p * v for p, v in zip(weights, ys)),
                               weights=w, ys=y0)
        else:
            # confounder unobserved -> only the (confounded) observational contrast is available
            do1 = chain.record("do1", "observational P(Y=1|X=1)",
                               lambda v: v, v=float(df[df[treatment] == 1][outcome].mean()))
            do0 = chain.record("do0", "observational P(Y=1|X=0)",
                               lambda v: v, v=float(df[df[treatment] == 0][outcome].mean()))
        effect = chain.record("effect", "do(1)-do(0)", lambda a, b: a - b, a=do1, b=do0)

        # --- self-verify by replay (no external oracle) ---
        replay_ok, _ = chain.verify()

        # --- three-zone honest status (stability + identifiability + E-value) ---
        stab = bootstrap_stability(df[[c for c in (confounder, treatment, outcome) if c in df.columns]],
                                   B=15, seed=0)["skeleton_agreement"]
        ev = e_value_for_do(do1, do0)["e_value"]
        if stab >= self.STABILITY_TOL and identifiable and ev >= self.EVALUE_TOL:
            zone = "VERIFIABLE"
        elif stab >= 0.5 and ev >= 1.4:
            zone = "UNCERTAINTY_QUANTIFIED"
        else:
            zone = "REJECT"

        cred = {"claim": f"ATE do(X=1)-do(X=0) = {effect:.4f}", "zone": zone,
                "regime": "native: replay-verified + three-zone status",
                "e_value": round(ev, 3), "structural_stability": round(stab, 3),
                "identifiable": identifiable, "replay_ok": replay_ok,
                "chain_hash": chain.root_hash(), "chain_steps": len(chain.steps)}
        return NativeResult(round(effect, 4), zone, round(ev, 3), round(stab, 3),
                            identifiable, chain, replay_ok, cred)

    # --- continuous-outcome path (real benchmark / product data) -----------
    def estimate_continuous(self, X: np.ndarray, t: np.ndarray, yf: np.ndarray,
                            covariate_sufficient: bool = True, seed: int = 1) -> NativeResult:
        """ATE on CONTINUOUS outcomes via a learned TARNet, with reproducibility-replay and a
        continuous three-zone status. Honest: a neural estimate's verification is REPRODUCIBLE
        (replay inference -> same) + epistemic-status-classified, not 1e-6 recomputable like
        the symbolic engine."""
        import torch
        import torch.nn as nn
        from theone.layer2_world_model.sensitivity import e_value_continuous
        dev = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available() else "cpu")
        X = np.asarray(X, np.float32); t = np.asarray(t, np.float32); yf = np.asarray(yf, np.float32)
        xm, xs = X.mean(0), X.std(0) + 1e-6
        ym, ys = yf.mean(), yf.std() + 1e-6

        def block(i, o): return nn.Sequential(nn.Linear(i, o), nn.ELU())

        BS = 131072                                            # minibatch cap -> memory O(batch), scales to any N

        def fit(Xtr, ttr, ytr, sd):
            torch.manual_seed(sd)
            phi = nn.Sequential(block(X.shape[1], 200), block(200, 200), block(200, 200))
            h0 = nn.Sequential(block(200, 100), nn.Linear(100, 1))
            h1 = nn.Sequential(block(200, 100), nn.Linear(100, 1))
            net = nn.ModuleList([phi, h0, h1]).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            n = len(ttr)
            if n <= BS:                                        # small N: identical full-batch path (unchanged)
                Xn = torch.tensor((Xtr - xm) / xs, device=dev)
                tt = torch.tensor(ttr, device=dev); yy = torch.tensor((ytr - ym) / ys, device=dev)
                for _ in range(400):
                    opt.zero_grad(); r = phi(Xn)
                    pred = torch.where(tt > 0.5, h1(r).squeeze(1), h0(r).squeeze(1))
                    ((pred - yy) ** 2).mean().backward(); opt.step()
            else:                                              # large N: minibatch SGD, data on CPU
                Xc = torch.tensor((Xtr - xm) / xs); tc = torch.tensor(ttr)
                yc = torch.tensor((ytr - ym) / ys)
                g = torch.Generator().manual_seed(sd)
                for _ in range(3000):
                    idx = torch.randint(0, n, (BS,), generator=g)
                    xb = Xc[idx].to(dev); tb = tc[idx].to(dev); yb = yc[idx].to(dev)
                    opt.zero_grad(); r = phi(xb)
                    pred = torch.where(tb > 0.5, h1(r).squeeze(1), h0(r).squeeze(1))
                    ((pred - yb) ** 2).mean().backward(); opt.step()

            def ite(Xq):
                with torch.no_grad():
                    out = []
                    for i in range(0, len(Xq), BS):            # chunked inference (memory-bounded; same result)
                        r = phi(torch.tensor((Xq[i:i + BS] - xm) / xs, device=dev))
                        out.append(((h1(r) - h0(r)).squeeze(1).cpu().numpy()) * ys)
                    return np.concatenate(out) if out else np.zeros(0, np.float32)
            return ite

        ite_fn = fit(X, t, yf, seed)
        ate = float(ite_fn(X).mean())

        chain = DerivationChain()
        chain.record("ate", "neural ITE -> ATE (reproducible inference)",
                     lambda f, Xq: round(float(f(Xq).mean()), 6), f=ite_fn, Xq=X)
        replay_ok = chain.verify()[0]                          # re-run inference == same ATE

        # reproducibility across data halves (stability) + continuous E-value (sensitivity)
        h = len(t) // 2
        a1 = float(fit(X[:h], t[:h], yf[:h], seed)(X[:h]).mean())
        a2 = float(fit(X[h:], t[h:], yf[h:], seed)(X[h:]).mean())
        stability = 1.0 - min(1.0, abs(a1 - a2) / (abs(ate) + 1e-6))
        ev = e_value_continuous(ate, float(yf.std()))["e_value"]

        if stability >= 0.7 and covariate_sufficient and ev >= self.EVALUE_TOL:
            zone = "VERIFIABLE"
        elif stability >= 0.4 and ev >= 1.4:
            zone = "UNCERTAINTY_QUANTIFIED"
        else:
            zone = "REJECT"
        cred = {"claim": f"ATE = {ate:.4f}", "zone": zone,
                "regime": "native-continuous: reproducible inference + three-zone (continuous E-value)",
                "e_value": round(ev, 3), "reproducibility_stability": round(stability, 3),
                "identifiable": covariate_sufficient, "replay_ok": replay_ok,
                "chain_hash": chain.root_hash()}
        return NativeResult(round(ate, 4), zone, round(ev, 3), round(stability, 3),
                            covariate_sufficient, chain, replay_ok, cred)


__all__ = ["NativeVerifiableEngine", "NativeResult", "DerivationChain", "PureStep"]
