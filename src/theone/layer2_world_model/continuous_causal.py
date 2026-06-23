"""L1->L2 ContinuousCausalLayer — verifiable de-confounded do() on a continuous latent.

Where CausalLayer needs a discrete CPT graph, this layer consumes the CONTINUOUS
latent/proxy stream L1 produces and estimates do() through a learned backdoor
adjustment. It is the realized seam between perception (L1) and causal reasoning (L2),
prototyped by the native_causal_latent probes. Two gates:
  • admissibility: subset-spread bias indicator < tol AND split-half recompute < tol
    (else ABSTAIN — proxy-incompleteness bias too high to certify);
  • recomputability: the credential's value is reproduced by an independent split-half
    estimate.
The regime declares the honest limit: variance-bounded, bias-partially-certified.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer2_world_model.continuous_do import (
    do_estimate, subset_spread, recompute_gap,
)


class ContinuousCausalLayer(CredentialedLayer):
    name = "L2c_continuous_causal"
    layer_index = 2          # an L2 variant; sits at the causal tier
    SPREAD_TOL = 0.02
    RECOMPUTE_TOL = 0.01

    def process(self, inputs: Any) -> LayerVerdict:
        proxies = np.asarray(inputs["proxies"], dtype=float)
        x = np.asarray(inputs["treatment"], dtype=float)
        y = np.asarray(inputs["outcome"], dtype=float)

        do_hat = do_estimate(proxies, x, y)
        spread = subset_spread(proxies, x, y)
        gap = recompute_gap(proxies, x, y)

        if np.isnan(spread):
            return LayerVerdict.abstain(
                self.name, "single proxy: proxy-completeness (bias) is uncheckable")
        if spread > self.SPREAD_TOL:
            return LayerVerdict.abstain(
                self.name, f"proxy-subset spread {spread:.3f} > {self.SPREAD_TOL} "
                           f"(proxy-incompleteness bias too high to certify)")
        if gap > self.RECOMPUTE_TOL:
            return LayerVerdict.abstain(
                self.name, f"split-half recompute gap {gap:.3f} > {self.RECOMPUTE_TOL}")

        cred = Credential(
            self.name, claim="P(outcome=1 | do(treatment=1)) via learned continuous-latent adjustment",
            value=round(do_hat, 9),
            regime="learned-latent de-confounding; variance-bounded, bias-partially-certified",
            recompute=lambda: round(do_estimate(proxies, x, y), 9),  # deterministic reproducibility
            tolerance=1e-9,
            evidence={"subset_spread": spread, "recompute_gap": gap,
                      "n": int(len(y)), "n_proxies": int(np.atleast_2d(proxies).shape[1]
                                                          if proxies.ndim > 1 else 1)})
        return LayerVerdict.answer(self.name, cred, value={**inputs, "do_value": do_hat})


__all__ = ["ContinuousCausalLayer"]
