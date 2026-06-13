"""Does the Lo Shu's defining feature (the 2 diagonal sum=15 constraints) add
error-detection power over plain row+col checksums? Full enumeration, no deps.

Prompted by 何老师's analysis (RESPONSE_TO_HE_TEACHER.md). Tests the ACTUAL magic
property (multi-directional conservation), not a value-independent proxy.

Result: diagonals reject 64/72 (89%) of the corrupting permutations that survive
row+col -- BUT for ordinary single/local errors (2-cell swaps) row+col already
catch ALL of them and diagonals add ZERO. So the diagonal redundancy only earns
its keep against an exotic threat model (sum-preserving global rearrangement).

Run: python experiments/luoshu_routing/diagonal_value.py
"""
import itertools


def sums_ok(g, with_diag):
    rows = all(sum(g[i]) == 15 for i in range(3))
    cols = all(g[0][j] + g[1][j] + g[2][j] == 15 for j in range(3))
    if not with_diag:
        return rows and cols
    d1 = g[0][0] + g[1][1] + g[2][2] == 15
    d2 = g[0][2] + g[1][1] + g[2][0] == 15
    return rows and cols and d1 and d2


def main():
    semimagic = magic = 0
    for p in itertools.permutations(range(1, 10)):
        g = [list(p[0:3]), list(p[3:6]), list(p[6:9])]
        if sums_ok(g, with_diag=False):
            semimagic += 1
            if sums_ok(g, with_diag=True):
                magic += 1
    print(f"9! arrangements = 362880")
    print(f"pass row+col (semimagic)      = {semimagic}")
    print(f"pass row+col+diag (Lo Shu)    = {magic}")
    print(f"diagonals reject {semimagic - magic}/{semimagic} = "
          f"{round(100*(semimagic-magic)/semimagic)}% of row+col survivors")

    base = [4, 9, 2, 3, 5, 7, 8, 1, 6]
    rc = diag_only = 0
    swaps = list(itertools.combinations(range(9), 2))
    for i, j in swaps:
        g = base[:]; g[i], g[j] = g[j], g[i]
        M = [g[0:3], g[3:6], g[6:9]]
        if not sums_ok(M, with_diag=False):
            rc += 1
        elif not sums_ok(M, with_diag=True):
            diag_only += 1
    print(f"\n2-cell swaps = {len(swaps)}: caught by row+col = {rc}, "
          f"ONLY by diagonals = {diag_only}, undetected = {len(swaps)-rc-diag_only}")
    print("Verdict: diagonal redundancy helps only vs sum-preserving global "
          "rearrangement; for local errors row+col suffice.")


if __name__ == "__main__":
    main()
