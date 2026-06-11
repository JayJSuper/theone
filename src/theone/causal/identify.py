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


def find_adjustment_set(g: CausalGraph, X: str, Y: str):
    """Smallest valid backdoor set (brute force; v0.1 graphs are small).
    Returns None when no valid set exists (not identifiable via backdoor);
    callers must surface None explicitly - 'cannot say' is a first-class answer."""
    candidates = [v for v in g.variables
                  if v not in (X, Y) and v not in _descendants(g, X)]
    for size in range(0, len(candidates) + 1):
        for Z in itertools.combinations(candidates, size):
            if check_backdoor(g, X, Y, set(Z)):
                return set(Z)
    return None
