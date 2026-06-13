"""Backdoor identification. CC-04. Textbook Pearl semantics on small discrete DAGs."""
from __future__ import annotations
import itertools
import networkx as nx
from .graph import CausalGraph


def _descendants(g: CausalGraph, x: str) -> set:
    return nx.descendants(g.nx, x)


def backdoor_paths(g: CausalGraph, X: str, Y: str) -> list:
    """All simple paths X..Y in the skeleton whose first edge points INTO X."""
    ug = g.nx.to_undirected()
    out = []
    for path in nx.all_simple_paths(ug, X, Y):
        if len(path) >= 2 and g.nx.has_edge(path[1], path[0]):  # edge into X
            out.append(path)
    return out


def _is_collider(g: CausalGraph, path: list, i: int) -> bool:
    a, b, c = path[i - 1], path[i], path[i + 1]
    return g.nx.has_edge(a, b) and g.nx.has_edge(c, b)


def is_blocked(path: list, Z: set, g: CausalGraph) -> bool:
    """d-separation blocking of a single path given Z."""
    Z = set(Z)
    for i in range(1, len(path) - 1):
        node = path[i]
        if _is_collider(g, path, i):
            opened = node in Z or any(d in Z for d in _descendants(g, node))
            if not opened:
                return True  # unconditioned collider blocks
        else:
            if node in Z:
                return True  # conditioned non-collider blocks
    return False


def check_backdoor(g: CausalGraph, X: str, Y: str, Z: set) -> bool:
    """Backdoor criterion: Z has no descendant of X AND blocks every backdoor path."""
    Z = set(Z)
    if Z & _descendants(g, X):
        return False
    return all(is_blocked(p, Z, g) for p in backdoor_paths(g, X, Y))


def find_adjustment_set(g: CausalGraph, X: str, Y: str, observed=None):
    """Smallest valid backdoor set among OBSERVED variables (brute force; v0.1
    graphs are small). Returns None when no valid observed set exists (not
    identifiable via backdoor); callers must surface None explicitly - 'cannot
    say' is a first-class answer."""
    obs = set(g.variables) if observed is None else set(observed)
    candidates = [v for v in g.variables
                  if v not in (X, Y) and v not in _descendants(g, X) and v in obs]
    for size in range(0, len(candidates) + 1):
        for Z in itertools.combinations(candidates, size):
            if check_backdoor(g, X, Y, set(Z)):
                return set(Z)
    return None


# ---------------------------------------------------------------------------
# Front-door criterion (Q-C14): usable when the X<->Y confounder is UNOBSERVED
# but an observed mediator M intercepts all directed X->Y paths.
# ---------------------------------------------------------------------------
def _directed_paths(g: CausalGraph, X: str, Y: str) -> list:
    return list(nx.all_simple_paths(g.nx, X, Y))


def check_frontdoor(g: CausalGraph, X: str, Y: str, M: set) -> bool:
    """Pearl front-door criterion for mediator set M relative to (X,Y):
    (1) M intercepts every directed path X->Y;
    (2) no unblocked backdoor path from X to M (P(M|do X)=P(M|X));
    (3) every backdoor path from M to Y is blocked by X."""
    M = set(M)
    if not M or M & {X, Y}:
        return False
    # (1) every directed X->Y path hits M
    if any(not (set(p[1:-1]) & M) for p in _directed_paths(g, X, Y)):
        return False
    # (2) no backdoor X->m unblocked by empty set, for each m
    for m in M:
        if not all(is_blocked(p, set(), g) for p in backdoor_paths(g, X, m)):
            return False
    # (3) backdoor m->Y blocked by {X}
    for m in M:
        if not check_backdoor(g, m, Y, {X}):
            return False
    return True


def find_frontdoor_set(g: CausalGraph, X: str, Y: str, observed=None):
    """Smallest observed mediator set satisfying the front-door criterion, or None."""
    obs = set(g.variables) if observed is None else set(observed)
    cand = [v for v in g.variables if v not in (X, Y) and v in obs]
    for size in range(1, len(cand) + 1):
        for M in itertools.combinations(cand, size):
            if check_frontdoor(g, X, Y, set(M)):
                return set(M)
    return None


# ---------------------------------------------------------------------------
# Single instrumental variable (Q-C14): usable when X<->Y confounder unobserved
# and an observed Z is relevant to X yet reaches Y only through X.
# ---------------------------------------------------------------------------
def check_instrument(g: CausalGraph, X: str, Y: str, Z: str) -> bool:
    """Graphical IV conditions for a single instrument Z (W=empty):
    (relevance) Z is an ancestor of X (Z d-connected to X);
    (exclusion+exogeneity) in G with X's OUTGOING edges removed, Z is
    d-separated from Y; (Z is not a descendant of X)."""
    if Z in (X, Y) or Z in _descendants(g, X):
        return False
    if X not in nx.descendants(g.nx, Z):            # relevance: Z -> ... -> X
        return False
    g_xbar = g.nx.copy()
    g_xbar.remove_edges_from(list(g.nx.out_edges(X)))  # cut X's outgoing edges
    try:
        return nx.is_d_separator(g_xbar, {Z}, {Y}, set())
    except AttributeError:                          # older networkx
        return nx.d_separated(g_xbar, {Z}, {Y}, set())


def find_instrument(g: CausalGraph, X: str, Y: str, observed=None):
    """An observed single instrument for X->Y, or None."""
    obs = set(g.variables) if observed is None else set(observed)
    for Z in g.variables:
        if Z in obs and check_instrument(g, X, Y, Z):
            return Z
    return None


# ---------------------------------------------------------------------------
# Unified identification (Q-C14 priority: backdoor > front-door > IV > refuse).
# Front-door/IV carry UNVERIFIABLE-FROM-DATA assumption flags (Jack's caveat).
# ---------------------------------------------------------------------------
def identify_effect(g: CausalGraph, X: str, Y: str, observed=None) -> dict:
    bd = find_adjustment_set(g, X, Y, observed)
    if bd is not None:
        return {"identifiable": True, "strategy": "backdoor",
                "adjustment_set": sorted(bd), "assumptions": []}
    fd = find_frontdoor_set(g, X, Y, observed)
    if fd is not None:
        return {"identifiable": True, "strategy": "front_door",
                "mediator_set": sorted(fd),
                "assumptions": ["front-door assumes no unobserved confounding "
                                "between mediator and outcome — NOT verifiable "
                                "from data, needs domain knowledge",
                                "near-violations are silent: a 0.05/0.3-strength "
                                "hidden mediator-outcome confounder biases the "
                                "estimate by ~0.003/0.082 with NO warning "
                                "(quantified probe, registry NOTE-002)"]}
    iv = find_instrument(g, X, Y, observed)
    if iv is not None:
        return {"identifiable": True, "strategy": "instrumental_variable",
                "instrument": iv,
                "assumptions": ["IV assumes the exclusion restriction (instrument "
                                "affects outcome only through treatment) — NOT "
                                "verifiable from data, needs domain knowledge"]}
    return {"identifiable": False, "strategy": None,
            "reason": "no backdoor / front-door / single-IV identification exists "
                      "among observed variables"}
