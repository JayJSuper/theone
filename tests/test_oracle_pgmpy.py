"""Third-party oracle cross-validation (dev-only): our frozen T1 engine vs
pgmpy's independent inference. Same philosophy as the credential verifier —
our own math, recomputed by code we didn't write.

Skipped automatically when pgmpy isn't installed (it's a dev extra, not a
runtime dependency — the engine itself stays stdlib+numpy+networkx)."""
import itertools
import pytest

pgmpy = pytest.importorskip("pgmpy")
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from theone.types import Variable
from theone.causal.graph import CausalGraph
from theone.causal.engine import InterventionEngine


def t1_graph():
    g = CausalGraph()
    for n in ("U", "X", "Y"):
        g.add_variable(Variable(n))
    g.add_edge("U", "X"); g.add_edge("U", "Y"); g.add_edge("X", "Y")
    g.set_cpt("U", {(): {1: 0.5, 0: 0.5}})
    g.set_cpt("X", {(1,): {1: 0.8, 0: 0.2}, (0,): {1: 0.2, 0: 0.8}})
    g.set_cpt("Y", {(1, 1): {1: 0.9, 0: 0.1}, (0, 1): {1: 0.5, 0: 0.5},
                    (1, 0): {1: 0.6, 0: 0.4}, (0, 0): {1: 0.2, 0: 0.8}})
    return g


def to_pgmpy(g: CausalGraph) -> DiscreteBayesianNetwork:
    """Translate our graph into pgmpy's representation (independent code path)."""
    m = DiscreteBayesianNetwork([(a, b) for a, b in g.nx.edges()])
    for v in g.variables:
        parents = list(g.parent_order(v))
        if not parents:
            p1 = g.cpt(v)[()][1]
            cpd = TabularCPD(v, 2, [[1 - p1], [p1]])
        else:
            cols = []
            # pgmpy column order: parent states iterate with LAST parent fastest,
            # state order [0, 1]
            for combo in itertools.product([0, 1], repeat=len(parents)):
                key = tuple(combo)
                p1 = g.cpt(v)[key][1]
                cols.append([1 - p1, p1])
            arr = list(map(list, zip(*cols)))
            cpd = TabularCPD(v, 2, arr, evidence=parents,
                             evidence_card=[2] * len(parents))
        m.add_cpds(cpd)
    m.check_model()
    return m


class TestPgmpyOracle:
    def test_t1_observation_matches_oracle(self):
        """P(Y=1|X=1) frozen truth 0.82: our engine vs pgmpy VariableElimination."""
        g = t1_graph()
        ours = InterventionEngine(g).query_observation("Y", 1, {"X": 1}).value
        infer = VariableElimination(to_pgmpy(g))
        theirs = float(infer.query(["Y"], evidence={"X": 1}).values[1])
        assert ours == pytest.approx(0.82, abs=1e-9)
        assert ours == pytest.approx(theirs, abs=1e-9)

    def test_t1_intervention_matches_oracle_surgery(self):
        """P(Y=1|do(X=1)) frozen truth 0.70: graph surgery cross-checked by
        building the mutilated network in pgmpy ourselves (edges into X removed,
        X clamped via virtual evidence on the surgered net)."""
        g = t1_graph()
        ours = InterventionEngine(g).query_intervention("Y", 1, {"X": 1}).value
        # surgery in pgmpy: remove U->X, set X's CPT to point mass at 1
        m = DiscreteBayesianNetwork([("U", "Y"), ("X", "Y")])
        m.add_cpds(TabularCPD("U", 2, [[0.5], [0.5]]),
                   TabularCPD("X", 2, [[0.0], [1.0]]))
        cols = []
        for combo in itertools.product([0, 1], repeat=2):  # parents (U, X)
            key = tuple(combo)
            p1 = g.cpt("Y")[key][1]
            cols.append([1 - p1, p1])
        order = list(g.parent_order("Y"))
        m.add_cpds(TabularCPD("Y", 2, list(map(list, zip(*cols))),
                              evidence=order, evidence_card=[2, 2]))
        m.check_model()
        theirs = float(VariableElimination(m).query(["Y"]).values[1])
        assert ours == pytest.approx(0.70, abs=1e-9)
        assert ours == pytest.approx(theirs, abs=1e-9)
