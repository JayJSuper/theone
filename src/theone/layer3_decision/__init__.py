"""Layer 3 · active-inference decision — the action center.

Honest scope (fusion mandate): the VFE engine minimizes prediction error under a
generative model and the active-inference loop keeps a running cycle's free energy
low. The credential certifies CONVERGENCE (gradient descent reaching the closed-form
optimum, monotonically), not 'autonomy'. DecisionLayer puts this on the spine.
"""
from theone.layer3_decision.vfe_engine import VFEEngine
from theone.layer3_decision.active_inference import ActiveInferenceLoop
from theone.layer3_decision.decision_layer import DecisionLayer
from theone.layer3_decision.cognitive_updater import CognitiveUpdater

__all__ = ["VFEEngine", "ActiveInferenceLoop", "DecisionLayer", "CognitiveUpdater"]
