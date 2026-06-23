"""Layer 2 · causal world model — the verified do-engine, re-homed onto the spine.

This layer does NOT reimplement causality: it wraps the frozen `InterventionEngine`
(exact graph-surgery do) and gates every answer with an independent pgmpy
recomputation (IPRG). The engine is the computation; pgmpy is the recomputability
gate; the regime field declares the honest limit (computation-exact, structure-assumed).
"""
import warnings as _warnings
# Silence pgmpy 1.1.x deprecation FutureWarnings (StructureScore re-export +
# HillClimbSearch relocation). pgmpy issues these with stacklevel pointing at the
# CALLER, so a module-based filter misses them — match the specific messages instead.
# Scoped to these exact pgmpy messages; our own warnings are untouched.
for _msg in (r".*HillClimbSearch is deprecated.*", r".*StructureScore.*deprecated.*"):
    _warnings.filterwarnings("ignore", message=_msg, category=FutureWarning)

from theone.layer2_world_model.causal_layer import CausalLayer
from theone.layer2_world_model.continuous_causal import ContinuousCausalLayer
from theone.layer2_world_model.discovery_layer import CausalDiscoveryLayer
from theone.layer2_world_model.sensitivity import e_value_for_do, e_value_rr
from theone.layer2_world_model.fit import fit_cpts
from theone.layer2_world_model.fit_layer import StructureFitLayer

__all__ = ["CausalLayer", "ContinuousCausalLayer", "CausalDiscoveryLayer",
           "StructureFitLayer", "fit_cpts", "e_value_for_do", "e_value_rr"]
