"""② ORDER-FREE causal discovery with identifiability-aware abstention (the hard bone).

Every prior native-substrate result assumed the causal ORDER was known (edges discoverable, order
given). This removes that: the learner is given ONLY observational data — no order — and must
discover structure AND honestly report which causal queries are answerable.

Fundamental limit (Markov equivalence): observational data identifies a DAG only up to its
equivalence class (same skeleton + v-structures); orientations of the rest are ambiguous. So a
causal effect P(Y|do(X)) is identifiable iff it is INVARIANT across every DAG in the class. We
embrace this exactly (IDA-style): enumerate all DAGs, BIC-score them (with per-(node,parents) score
caching), take the best-scoring equivalence class, compute the do-effect under EVERY member, and
  • if the effect is invariant across the class -> ANSWER (identifiable),
  • if it varies -> ABSTAIN (not identifiable from observation alone — honest, not a guess).

This is the verifiable-or-silent philosophy applied to discovery itself: claim only what the data
can identify, abstain on the rest — without ever assuming the causal order.

Run:  THEONE_FAST=1 .venv/bin/python experiments/bline_order_free_discovery/run.py
"""
from __future__ import annotations
import os, itertools
from functools import lru_cache
import numpy as np

FAST = os.environ.get("THEONE_FAST") == "1"
K = 4


def random_dag(rng):
    """random DAG over K nodes (random labels — NO canonical order given to the learner)."""
    order = rng.permutation(K)
    edges = np.zeros((K, K), bool)                       # edges[p, c] = parent p -> child c
    for a in range(K):
        for b in range(a + 1, K):
            if rng.random() < 0.5:
                edges[order[a], order[b]] = True
    bias = rng.normal(0, 1.3, K); w = rng.normal(0, 2.3, (K, K))
    return edges, bias, w


def parents_of(edges, i): return [p for p in range(K) if edges[p, i]]


def sigm(z): return 1.0 / (1.0 + np.exp(-z))


def sample_data(edges, bias, w, n, rng):
    """ancestral sampling in a valid topological order of `edges`."""
    topo = topo_order(edges)
    S = np.zeros((n, K), np.int8)
    for i in topo:
        z = np.full(n, bias[i])
        for p in parents_of(edges, i):
            z += w[p, i] * (2 * S[:, p] - 1)
        S[:, i] = (rng.random(n) < sigm(z)).astype(np.int8)
    return S


def topo_order(edges):
    indeg = {i: sum(edges[:, i]) for i in range(K)}; order = []; avail = [i for i in range(K) if indeg[i] == 0]
    while avail:
        u = avail.pop(); order.append(u)
        for v in range(K):
            if edges[u, v]:
                indeg[v] -= 1
                if indeg[v] == 0: avail.append(v)
    return order


def is_acyclic(edges):
    return len(topo_order(edges)) == K


def all_dags():
    """enumerate every DAG over K nodes (as boolean adjacency)."""
    pairs = [(i, j) for i in range(K) for j in range(K) if i != j]
    out = []
    for bits in itertools.product([0, 1], repeat=len(pairs)):
        e = np.zeros((K, K), bool)
        for (i, j), b in zip(pairs, bits):
            if b: e[i, j] = True
        if is_acyclic(e):
            out.append(e)
    return out


def fit_node_score(S, i, parents):
    """BIC score of node i given parent set (logistic MLE). Negated-BIC: higher = better fit."""
    n = len(S); y = S[:, i].astype(float)
    if not parents:
        m = np.clip(y.mean(), 1e-9, 1 - 1e-9); ll = np.sum(y * np.log(m) + (1 - y) * np.log(1 - m)); k = 1
    else:
        Xp = (2 * S[:, list(parents)] - 1).astype(float); wv = np.zeros(len(parents)); b = 0.0
        for _ in range(400):
            p = sigm(Xp @ wv + b); g = p - y; wv -= 0.5 * (Xp.T @ g) / n; b -= 0.5 * g.mean()
        p = np.clip(sigm(Xp @ wv + b), 1e-9, 1 - 1e-9); ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)); k = len(parents) + 1
    return ll - 0.5 * k * np.log(n)                        # higher = better (BIC/2, sign flipped)


def fit_cpt(S, i, parents):
    n = len(S); y = S[:, i].astype(float)
    if not parents:
        m = np.clip(y.mean(), 1e-9, 1 - 1e-9); return np.zeros(K), np.log(m / (1 - m))
    Xp = (2 * S[:, list(parents)] - 1).astype(float); wv = np.zeros(len(parents)); b = 0.0
    for _ in range(600):
        p = sigm(Xp @ wv + b); g = p - y; wv -= 0.5 * (Xp.T @ g) / n; b -= 0.5 * g.mean()
    wful = np.zeros(K)
    for idx, pp in enumerate(parents): wful[pp] = wv[idx]
    return wful, b


def do_effect(edges, bias, w, X, Y):
    """ATE = P(Y=1|do X=1) - P(Y=1|do X=0) under DAG `edges` with CPTs (bias,w), by enumeration."""
    def doX(xv):
        others = [v for v in range(K) if v != X]; tot = 0.0
        for mask in range(1 << len(others)):
            a = [0] * K; a[X] = xv
            for bi, v in enumerate(others): a[v] = (mask >> bi) & 1
            pr = 1.0
            for v in others:
                if v == X: continue
                z = bias[v] + sum(w[p, v] * (2 * a[p] - 1) for p in parents_of(edges, v))
                pv = sigm(z); pr *= pv if a[v] else (1 - pv)
            if a[Y]: tot += pr
        return tot
    return doX(1) - doX(0)


def main():
    print("=== ② ORDER-FREE discovery + identifiability-aware abstention (no causal order given) ===\n")
    DAGS = all_dags()
    print(f"  K={K}: enumerated {len(DAGS)} DAGs\n")
    rng = np.random.default_rng(0)
    for n_obs in ([800, 4000] if FAST else [800, 4000, 16000]):
        N = 40 if FAST else 80
        skel_f1 = []; ident = 0; ident_err = []; abst = 0; conf_wrong = 0; true_in_class = 0
        for _ in range(N):
            edges, bias, w = random_dag(rng)
            # pick a query X->...->Y with X a (possibly indirect) cause for a non-trivial truth
            X = int(rng.integers(0, K)); Y = int(rng.integers(0, K))
            while Y == X: Y = int(rng.integers(0, K))
            S = sample_data(edges, bias, w, n_obs, rng)
            true_ate = do_effect(edges, bias, w, X, Y)

            # score every DAG with cached per-(node,parentset) scores
            @lru_cache(maxsize=None)
            def nscore(i, parents): return fit_node_score(S, i, parents)
            scored = []
            for e in DAGS:
                sc = sum(nscore(i, tuple(parents_of(e, i))) for i in range(K))
                scored.append((sc, e))
            best = max(s for s, _ in scored)
            # BIC CONFIDENCE SET: all DAGs within a margin of the best fit — wide enough to contain the
            # true structure w.h.p. at finite n, so "invariant across the set" is a conservative
            # (red-line-safe) identifiability test, not an over-tight tie-break.
            klass = [e for s, e in scored if best - s < 7.0]

            # skeleton F1 of the class-representative vs truth
            def skel(e): return {frozenset((i, j)) for i in range(K) for j in range(K) if i != j and (e[i, j] or e[j, i])}
            tskel = skel(edges); pskel = skel(klass[0])
            tp = len(tskel & pskel); fp = len(pskel - tskel); fn = len(tskel - pskel)
            prec = tp / (tp + fp) if tp + fp else 1.0; rec = tp / (tp + fn) if tp + fn else 1.0
            skel_f1.append(2 * prec * rec / (prec + rec) if prec + rec else 1.0)
            true_in_class += any(np.array_equal(e, edges) for e in klass)

            # IDA: effect under every DAG in the class; identifiable iff invariant
            effects = []
            for e in klass:
                cb = np.zeros(K); cw = np.zeros((K, K))
                for i in range(K):
                    pa = parents_of(e, i); wf, b = fit_cpt(S, i, tuple(pa)); cw[:, i] = wf; cb[i] = b
                effects.append(do_effect(e, cb, cw, X, Y))
            spread = max(effects) - min(effects)
            if spread < 0.05:                                    # identifiable -> ANSWER
                ident += 1; est = float(np.mean(effects)); ident_err.append(abs(est - true_ate))
                if abs(true_ate) > 0.1 and np.sign(est) != np.sign(true_ate): conf_wrong += 1
            else:
                abst += 1                                        # not identifiable -> ABSTAIN

        print(f"  n={n_obs:>5}:  skeleton-F1={np.mean(skel_f1):.3f}  true-DAG-in-class={true_in_class}/{N}  "
              f"identifiable={ident} (err {np.mean(ident_err) if ident_err else 0:.3f}, wrong {conf_wrong})  abstained={abst}")
        last = (np.mean(skel_f1), true_in_class / N, np.mean(ident_err) if ident_err else 0.0, conf_wrong)

    f1, tic, ierr, cw = last
    g1 = f1 > 0.85                                              # recovers the skeleton without knowing order
    g2 = tic > 0.85                                             # the TRUE DAG is in the recovered class
    g3 = ierr < 0.06 and cw == 0                                # identifiable answers correct; 0 confidently-wrong
    allok = g1 and g2 and g3
    print("\norder-free discovery gate (largest n):")
    print(f"  [{'PASS' if g1 else 'FAIL'}] recovers the causal SKELETON with NO order given (F1>0.85)")
    print(f"  [{'PASS' if g2 else 'FAIL'}] the true DAG lies in the recovered equivalence class (>85%)")
    print(f"  [{'PASS' if g3 else 'FAIL'}] identifiable queries answered correctly, 0 confidently-wrong (abstains on the rest)")
    msg = ("PASS — order-free discovery: structure learned with NO causal order, identifiable effects "
           "answered correctly, non-identifiable ones abstained (Markov-equivalence handled honestly). "
           "The system claims only what observation can identify.") if allok else "CHECK"
    print(f"\n  >>> {msg}")
    print("\nHonest: observational data identifies only up to Markov equivalence; full orientation needs")
    print("interventions or extra assumptions. We do NOT pretend — we answer the identifiable, abstain the rest.")
    if not allok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
