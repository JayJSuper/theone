"""L2 CausalDiscoveryLayer — structure discovery with an honest reliability gate.

The deepest-honesty layer (this is NOTE-004 made operational). It discovers a candidate
structure and ANSWERs only when the SKELETON is stable under bootstrap resampling
(finite-sample reliability it CAN check), while the credential's regime declares the
two limits it CANNOT check from observational data: edge orientation within the Markov
equivalence class, and latent confounding. Unstable skeletons ABSTAIN — a confidently
wrong structure never passes.
"""
from __future__ import annotations
from typing import Any
import pandas as pd

from theone.core.spine import CredentialedLayer, LayerVerdict, Credential
from theone.layer2_world_model.discovery import discover, bootstrap_stability


class CausalDiscoveryLayer(CredentialedLayer):
    name = "L2d_discovery"
    layer_index = 2
    STABILITY_TOL = 0.8

    def process(self, inputs: Any) -> LayerVerdict:
        data = inputs["data"]
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        B = int(inputs.get("B", 25))
        thr = float(inputs.get("stability_threshold", self.STABILITY_TOL))

        edges = discover(df)
        if not edges:
            return LayerVerdict.abstain(self.name, "no edges discovered (no detectable structure)")
        stab = bootstrap_stability(df, B=B, seed=inputs.get("seed", 0))
        skel_freq = stab["skeleton_freq"]
        edge_freq = stab["edge_freq"]
        agreement = stab["skeleton_agreement"]

        # gate 1: the whole skeleton must be reproduced often enough under resampling
        if agreement < thr:
            return LayerVerdict.abstain(
                self.name, f"skeleton unstable: only {agreement:.2f} of bootstrap resamples "
                           f"reproduce the discovered skeleton (< {thr}); n={len(df)} insufficient")

        # gate 2: every discovered link must itself be bootstrap-stable
        links = sorted({tuple(sorted((a, b))) for a, b in edges})
        link_stab = {lk: skel_freq.get(lk, 0.0) for lk in links}
        min_stab = min(link_stab.values())
        if min_stab < thr:
            weak = min(link_stab, key=link_stab.get)
            return LayerVerdict.abstain(
                self.name, f"skeleton unstable: link {weak} bootstrap freq "
                           f"{link_stab[weak]:.2f} < {thr} (n={len(df)} insufficient to discover reliably)")

        # orientation confidence within each stable link (max directed share)
        orient_conf = {}
        for (a, b) in links:
            f_ab = edge_freq.get((a, b), 0.0)
            f_ba = edge_freq.get((b, a), 0.0)
            tot = f_ab + f_ba
            orient_conf[(a, b)] = round(max(f_ab, f_ba) / tot, 3) if tot > 0 else 0.0

        cred = Credential(
            self.name, claim="causal skeleton discovered and sample-stable",
            value=str(links),
            regime=("structure-discovered (skeleton sample-stable); orientation within the "
                    "Markov-equivalence class AND latent confounding UNCERTIFIED (NOTE-004)"),
            recompute=lambda: str(sorted({tuple(sorted((a, b))) for a, b in discover(df)})),
            tolerance=0.0,
            evidence={"discovered_edges": edges, "skeleton_stability": link_stab,
                      "orientation_confidence": orient_conf, "n": int(len(df)), "B": B,
                      "min_skeleton_stability": round(min_stab, 3),
                      "skeleton_agreement": round(agreement, 3)})
        return LayerVerdict.answer(self.name, cred, value={**inputs, "discovered_edges": edges})


__all__ = ["CausalDiscoveryLayer"]
