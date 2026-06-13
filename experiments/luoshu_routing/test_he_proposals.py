"""Comprehensive test of 何老师's three proposed Luoshu experiments (A/B/C),
each run against HIS falsifiable criterion, with steelman variants. No torch
(Experiment A's training criterion noted separately); structural analysis used.

Findings (see RESPONSE_TO_HE_TEACHER.md §五):
  A  attention mask: value-independent (magic never used); his density is 80% not
     40%; even the steelman true-magic mask mixes WORSE (lower algebraic
     connectivity) than density-matched random sparse -> expected to underperform.
  B  memory path: no code bug (the double-indirection is the identity map), but
     ties sequential/random under uniform placement -> criterion fails.
  C  scheduling: no optimization value for arbitrary weights; 'all lines = 15'
     holds only for weights exactly 1..9 and is tautological (magic square's
     definition), not an algorithmic advantage; his code was a non-functional stub.

Run: python experiments/luoshu_routing/test_he_proposals.py
"""
import itertools
import numpy as np

LUO = [4, 9, 2, 3, 5, 7, 8, 1, 6]


# ----------------------------- Experiment B -----------------------------
def test_B(rng):
    pos_to_idx = {4: 0, 9: 1, 2: 2, 3: 3, 5: 4, 7: 5, 8: 6, 1: 7, 6: 8}
    # his read indexing: pos_to_idx[path[i]] == i for all i (identity) -> no bug
    identity = all(pos_to_idx[LUO[i]] == i for i in range(9))
    luo_order = [pos_to_idx[p] for p in LUO]
    seq_order = list(range(9))

    def avg_steps(order, n_items, trials=40000):
        tot = 0
        for _ in range(trials):
            filled = set(rng.choice(9, size=n_items, replace=False).tolist())
            start = int(rng.integers(0, 9))
            for s in range(9):
                if order[(start + s) % 9] in filled:
                    tot += s
                    break
        return tot / trials

    print("== Experiment B: memory addressing ==")
    print(f"  read-indexing identity (no bug): {identity}")
    for n in (1, 3, 5):
        a = avg_steps(luo_order, n); b = avg_steps(seq_order, n)
        c = avg_steps(list(rng.permutation(9)), n)
        print(f"  items={n}: luoshu={a:.3f} sequential={b:.3f} random={c:.3f} "
              f"-> criterion 'luoshu<random' {'HOLDS' if a < c - 0.02 else 'FAILS (tie)'}")


# ----------------------------- Experiment C -----------------------------
def line_sums(g):
    s = [sum(g[i]) for i in range(3)] + [g[0][j] + g[1][j] + g[2][j] for j in range(3)]
    return s + [g[0][0] + g[1][1] + g[2][2], g[0][2] + g[1][1] + g[2][0]]


def imbalance(weights, perm):
    g = [[weights[perm[3 * i + j]] for j in range(3)] for i in range(3)]
    ls = line_sums(g)
    return max(ls) - min(ls)


def test_C(rng):
    allperms = list(itertools.permutations(range(9)))
    luo_perm = list(np.argsort(np.argsort(LUO)))  # place rank-sorted weights by luoshu magnitude
    gaps = []
    for _ in range(300):
        w = sorted(rng.integers(1, 20, size=9).tolist())
        opt = min(imbalance(w, p) for p in allperms)
        gaps.append(imbalance(w, luo_perm) - opt)
    print("\n== Experiment C: multi-constraint scheduling ==")
    print(f"  arbitrary weights: mean(luoshu_imbalance - optimal) = {np.mean(gaps):.2f} "
          f"(>0 => luoshu placement is not optimal / no value)")
    w = list(range(1, 10))
    magic_perm = [w.index(v) for v in LUO]
    print(f"  weights=1..9: luoshu imbalance = {imbalance(w, magic_perm)} "
          f"(=0 BUT tautological: magic square is defined as all-lines-15)")


# ----------------------------- Experiment A -----------------------------
def mask_his(arr):
    coord = {v: (i // 3, i % 3) for i, v in enumerate(arr)}
    M = np.zeros((9, 9), bool)
    for i, pi in enumerate(range(1, 10)):
        for j, pj in enumerate(range(1, 10)):
            ri, ci = coord[pi]; rj, cj = coord[pj]
            M[i, j] = (ri == rj) or (ci == cj) or (ri - ci == rj - cj) or (ri + ci == rj + cj)
    return M


def mask_true_magic(arr):
    coord = {v: (i // 3, i % 3) for i, v in enumerate(arr)}
    main_d = {(0, 0), (1, 1), (2, 2)}; anti_d = {(0, 2), (1, 1), (2, 0)}
    M = np.zeros((9, 9), bool)
    for i, pi in enumerate(range(1, 10)):
        for j, pj in enumerate(range(1, 10)):
            a, b = coord[pi], coord[pj]
            same = (a[0] == b[0]) or (a[1] == b[1])
            diag = (a in main_d and b in main_d) or (a in anti_d and b in anti_d)
            M[i, j] = same or diag
    return M


def alg_conn(M):
    A = M.astype(float).copy(); np.fill_diagonal(A, 0)
    L = np.diag(A.sum(1)) - A
    return float(np.sort(np.linalg.eigvalsh(L))[1])


def test_A(rng):
    print("\n== Experiment A: sparse attention mask ==")
    # value-independence: same structure for any arrangement
    e_set = set()
    for arr in (LUO, list(range(1, 10)), [1, 3, 2, 4, 5, 6, 7, 9, 8]):
        e_set.add(int(mask_his(arr).sum()))
    print(f"  value-independence: edge-count across luoshu/natural/nonmagic = {e_set} "
          f"({'IDENTICAL -> magic property unused' if len(e_set) == 1 else 'differs'})")
    his = mask_his(LUO); tm = mask_true_magic(LUO)
    print(f"  his mask: density={his.sum()/81:.2f} (he claimed ~0.40), λ2={alg_conn(his):.2f}")
    print(f"  steelman true-magic mask: density={tm.sum()/81:.2f}, λ2={alg_conn(tm):.2f}")
    # density-matched random baseline (exact edge count, symmetric)
    target_edges = int(tm.sum())
    l2 = []
    upper = [(i, j) for i in range(9) for j in range(i + 1, 9)]
    for _ in range(500):
        M = np.zeros((9, 9), bool); np.fill_diagonal(M, True)
        need = (target_edges - 9) // 2
        chosen = rng.choice(len(upper), size=need, replace=False)
        for c in chosen:
            i, j = upper[c]; M[i, j] = M[j, i] = True
        l2.append(alg_conn(M))
    print(f"  density-matched random ({target_edges} edges): λ2 mean={np.mean(l2):.2f} "
          f"std={np.std(l2):.2f}")
    print(f"  -> true-magic λ2={alg_conn(tm):.2f} vs random {np.mean(l2):.2f}: "
          f"{'Luoshu mixes WORSE than random (structural disadvantage)' if alg_conn(tm) < np.mean(l2) else 'Luoshu >= random'}")


def main():
    rng = np.random.default_rng(20260613)
    test_B(rng)
    test_C(rng)
    test_A(rng)


if __name__ == "__main__":
    main()
