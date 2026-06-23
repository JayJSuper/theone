"""Independent P-Recompute Gate (IPRG) for Layer 2 — a pgmpy re-derivation of do(),
promoted from the verified oracle-crosscheck pattern to a first-class package citizen.

The engine (`theone.causal.engine`) computes do() by graph surgery over the full
joint. This module recomputes the SAME quantity by an INDEPENDENT implementation
(pgmpy variable elimination on a surgically modified network). Agreement to <1e-6 is
the project's root trust signal: we believe a number only when two unrelated engines
reproduce it. (Frozen evidence: 1207 SCMs, max gap <5e-7.)
"""
from __future__ import annotations
import itertools

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def to_pgmpy(g) -> DiscreteBayesianNetwork:
    """Translate a CausalGraph (binary nodes) into a pgmpy network. Independent of
    the engine's inference code (shares only the CPT data, never the algorithm)."""
    nodes = list(g.variables)
    edges = [(p, v) for v in nodes for p in g.parent_order(v)]
    m = DiscreteBayesianNetwork(edges)
    for v in nodes:
        if v not in m.nodes():
            m.add_node(v)
    cpds = []
    for v in nodes:
        order = list(g.parent_order(v))
        sn = {v: [0, 1]}
        for p in order:
            sn[p] = [0, 1]
        if not order:
            p1 = g.cpt(v)[()][1]
            cpds.append(TabularCPD(v, 2, [[1 - p1], [p1]], state_names={v: [0, 1]}))
        else:
            combos = list(itertools.product([0, 1], repeat=len(order)))
            r1 = [g.cpt(v)[c][1] for c in combos]
            r0 = [g.cpt(v)[c][0] for c in combos]
            cpds.append(TabularCPD(v, 2, [r0, r1], evidence=order,
                                   evidence_card=[2] * len(order), state_names=sn))
    m.add_cpds(*cpds)
    assert m.check_model()
    return m


def pgmpy_do1(g, x: str, y: str) -> float:
    """do(x=1): independent surgery — drop edges into x, pin x=1, query P(y=1)."""
    m = to_pgmpy(g)
    for p in list(g.parent_order(x)):
        m.remove_edge(p, x)
    m.remove_cpds(m.get_cpds(x))
    m.add_cpds(TabularCPD(x, 2, [[0.0], [1.0]], state_names={x: [0, 1]}))
    assert m.check_model()
    return float(VariableElimination(m).query([y], show_progress=False).values[1])


def pgmpy_do0(g, x: str, y: str) -> float:
    """do(x=0): symmetric, for interventional ATE recomputation."""
    m = to_pgmpy(g)
    for p in list(g.parent_order(x)):
        m.remove_edge(p, x)
    m.remove_cpds(m.get_cpds(x))
    m.add_cpds(TabularCPD(x, 2, [[1.0], [0.0]], state_names={x: [0, 1]}))
    assert m.check_model()
    return float(VariableElimination(m).query([y], show_progress=False).values[1])


__all__ = ["to_pgmpy", "pgmpy_do1", "pgmpy_do0"]
