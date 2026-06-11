"""CausalGraph: discrete DAG with CPTs. CC-03.
Frozen ruling (registry): construction-time VALIDATION replaces normalization.
Any probability outside [0,1] or distribution not summing to 1 is rejected with
GraphValidationError - never min-max rescaled (rescaling destroys cross-graph
comparability and locality)."""
from __future__ import annotations
import hashlib
import itertools
import json
import networkx as nx
from ..types import Variable, GraphValidationError

_SUM_TOL = 1e-9


class CausalGraph:
    def __init__(self) -> None:
        self._nx = nx.DiGraph()
        self._vars: dict[str, Variable] = {}
        self._cpt: dict[str, dict[tuple, dict]] = {}

    # -- construction -------------------------------------------------
    def add_variable(self, v: Variable) -> None:
        if v.name in self._vars:
            raise GraphValidationError(f"duplicate variable {v.name}")
        self._vars[v.name] = v
        self._nx.add_node(v.name)

    def add_edge(self, parent: str, child: str) -> None:
        for n in (parent, child):
            if n not in self._vars:
                raise GraphValidationError(f"unknown variable {n}")
        self._nx.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self._nx):
            self._nx.remove_edge(parent, child)
            raise GraphValidationError(f"edge {parent}->{child} creates a cycle")

    def set_cpt(self, var: str, cpt: dict) -> None:
        """cpt: {parent_value_tuple: {state: prob}}; parent order = sorted(parents).
        Roots use key ()."""
        if var not in self._vars:
            raise GraphValidationError(f"unknown variable {var}")
        fixed: dict[tuple, dict] = {}
        for key, dist in cpt.items():
            kk: tuple = key if isinstance(key, tuple) else (key,)
            total = 0.0
            for state, p in dist.items():
                p = float(p)
                if not (0.0 <= p <= 1.0):
                    raise GraphValidationError(
                        f"P({var}={state}|{kk})={p} outside [0,1]; "
                        "validation replaces normalization (frozen ruling)")
                total += p
            if abs(total - 1.0) > _SUM_TOL:
                raise GraphValidationError(
                    f"CPT row for {var}|{kk} sums to {total}, not 1")
            fixed[kk] = dict(dist)
        self._cpt[var] = fixed

    # -- accessors -----------------------------------------------------
    @property
    def variables(self) -> list:
        return list(nx.topological_sort(self._nx))

    def states(self, var: str) -> tuple:
        return self._vars[var].states

    def parents(self, var: str) -> set:
        return set(self._nx.predecessors(var))

    def parent_order(self, var: str) -> tuple:
        return tuple(sorted(self.parents(var)))

    def cpt(self, var: str) -> dict:
        return self._cpt[var]

    @property
    def nx(self) -> nx.DiGraph:
        return self._nx

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self._nx)

    # -- validation ----------------------------------------------------
    def validate(self) -> None:
        if not self.is_dag():
            raise GraphValidationError("graph contains a cycle")
        for var in self._vars:
            if var not in self._cpt:
                raise GraphValidationError(f"missing CPT for {var}")
            pa = self.parent_order(var)
            combos = (list(itertools.product(*[self._vars[p].states for p in pa]))
                      if pa else [()])
            for c in combos:
                if c not in self._cpt[var]:
                    raise GraphValidationError(f"missing CPT row {var}|{pa}={c}")
                row = self._cpt[var][c]
                for s in self._vars[var].states:
                    if s not in row:
                        raise GraphValidationError(f"missing P({var}={s}|{pa}={c})")

    # -- identity --------------------------------------------------------
    def content_hash(self) -> str:
        payload = {
            "vars": {n: list(v.states) for n, v in sorted(self._vars.items())},
            "edges": sorted(map(list, self._nx.edges())),
            "cpt": {v: {str(k): {str(s): self._cpt[v][k][s]
                                 for s in sorted(self._cpt[v][k], key=str)}
                        for k in sorted(self._cpt[v], key=str)}
                    for v in sorted(self._cpt)},
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
