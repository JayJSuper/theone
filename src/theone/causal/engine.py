"""InterventionEngine - the F-1 fix lives here. CC-05.

Frozen semantics (T1, registry):
  observation : P(Y|evidence) via EXACT joint inference - conditioning automatically
                propagates along backdoor paths (posterior weighting of confounders).
  intervention: graph surgery - incoming edges of do-variables removed, do-values
                clamped; confounders keep their PRIOR distribution.
Generalized implementation over the full joint; no special-casing of any graph
(A8 mechanism guard enforces this in tests)."""
from __future__ import annotations
import itertools
from ..types import QueryResult, CompareResult, TheOneConfig
from .graph import CausalGraph


class InterventionEngine:
    def __init__(self, graph: CausalGraph) -> None:
        graph.validate()
        self.g = graph

    # -- joint enumeration ------------------------------------------------
    def _assignments(self):
        names = self.g.variables
        for combo in itertools.product(*[self.g.states(v) for v in names]):
            yield dict(zip(names, combo))

    def _prob_of(self, assign: dict, do: dict | None = None) -> float:
        """Joint probability of a full assignment; under `do`, clamped variables
        contribute factor 1 if matching the do-value else 0 (graph surgery)."""
        do = do or {}
        p = 1.0
        for v in self.g.variables:
            if v in do:
                if assign[v] != do[v]:
                    return 0.0
                continue  # clamped: no CPT factor (incoming edges severed)
            key = tuple(assign[x] for x in self.g.parent_order(v))
            p *= self.g.cpt(v)[key][assign[v]]
            if p == 0.0:
                return 0.0
        return p

    def _marginal(self, target: str, value, evidence: dict | None = None,
                  do: dict | None = None) -> float:
        evidence = evidence or {}
        num = den = 0.0
        for a in self._assignments():
            if any(a[k] != v for k, v in evidence.items()):
                continue
            p = self._prob_of(a, do=do)
            den += p
            if a[target] == value:
                num += p
        if den == 0.0:
            raise ZeroDivisionError("evidence has probability zero")
        return num / den

    # -- public queries ----------------------------------------------------
    def query_observation(self, target: str, value, given: dict) -> QueryResult:
        """P(target=value | given). Exact conditioning => posterior weighting of
        every ancestor, including confounders along backdoor paths (T1 frozen)."""
        p = self._marginal(target, value, evidence=given)
        return QueryResult(value=p, method="exact_joint_conditioning",
                           details={"given": dict(given),
                                    "graph_hash": self.g.content_hash()})

    def query_intervention(self, target: str, value, do: dict) -> QueryResult:
        """P(target=value | do(...)). Graph surgery => prior weighting of
        confounders (T1 frozen)."""
        p = self._marginal(target, value, do=do)
        return QueryResult(value=p, method="graph_surgery_do",
                           details={"do": dict(do),
                                    "graph_hash": self.g.content_hash()})

    # -- ATEs (binary treatment/outcome convention: effect on outcome==1) --
    def observational_ate(self, X: str, Y: str) -> float:
        return (self.query_observation(Y, 1, {X: 1}).value
                - self.query_observation(Y, 1, {X: 0}).value)

    def interventional_ate(self, X: str, Y: str) -> float:
        return (self.query_intervention(Y, 1, {X: 1}).value
                - self.query_intervention(Y, 1, {X: 0}).value)

    def compare(self, X: str, Y: str, config: TheOneConfig | None = None) -> CompareResult:
        """Numeric mode (exact inference): difference beyond tolerance.
        Finite-sample statistical mode (A7 frozen conjunction) lives in bench.eg."""
        cfg = config or TheOneConfig()
        obs, intv = self.observational_ate(X, Y), self.interventional_ate(X, Y)
        diff = obs - intv
        return CompareResult(
            obs_ate=obs, int_ate=intv,
            are_different=abs(diff) > max(cfg.numeric_tol, 1e-12),
            stats={"diff": diff, "mode": "exact_numeric",
                   "graph_hash": self.g.content_hash()})
