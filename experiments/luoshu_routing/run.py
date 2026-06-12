"""Luoshu routing benchmark (frozen prereg 48c54b4). Honest fair fight.
Run: python experiments/luoshu_routing/run.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
MAGIC = [[4, 9, 2], [3, 5, 7], [8, 1, 6]]
MAGIC_FLAT = [4, 9, 2, 3, 5, 7, 8, 1, 6]           # row-major; each value 1..9 unique
K = 9
NS = [90, 900, 9000]
SEEDS = list(range(20))


def _h(key: str) -> int:
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16)


# ---------------- routers (key, index) -> bucket in [0,9) ----------------
def r_luoshu(key, idx):
    return MAGIC_FLAT[_h(key) % K] - 1            # map magic value 1..9 -> 0..8


def r_modulo(key, idx):
    return idx % K                                # round-robin on arrival order


def r_random(key, idx, rng):
    return int(rng.integers(0, K))


class ConsistentHash:
    def __init__(self, vnodes=160):
        self.ring = sorted((int(hashlib.md5(f"{b}-{v}".encode()).hexdigest(), 16), b)
                           for b in range(K) for v in range(vnodes))
        self.keys = [r[0] for r in self.ring]

    def route(self, key):
        import bisect
        h = _h(key)
        i = bisect.bisect(self.keys, h) % len(self.ring)
        return self.ring[i][1]


def imbalance(loads):
    loads = np.asarray(loads, float)
    mean = loads.mean()
    return float(loads.max() / mean) if mean > 0 else 0.0


def gen_keys(mode, n, rng):
    """Return list of (key, row, col) for 2D-balance metric."""
    if mode == "uniform":
        ks = [f"u{int(rng.integers(0, 1 << 30))}" for _ in range(n)]
    elif mode == "skewed":                        # Zipf s=1.2 over n/10 hot keys
        m = max(10, n // 10)
        ranks = rng.zipf(1.2, size=n) % m
        ks = [f"z{r}" for r in ranks]
    else:                                          # sequential_burst
        ks = []
        v = int(rng.integers(0, 1 << 20))
        for _ in range(n):
            if rng.random() < 0.1:
                v = int(rng.integers(0, 1 << 20))
            else:
                v += 1
            ks.append(f"s{v}")
    out = []
    for i, k in enumerate(ks):
        out.append((k, _h(k) % 3, (_h(k) // 3) % 3))  # synthetic 2D labels
    return out


def run_router(name, keys, rng):
    loads = np.zeros(K)
    row2d = np.zeros(3); col2d = np.zeros(3)
    ch = ConsistentHash() if name == "consistent_hash" else None
    for idx, (k, r, c) in enumerate(keys):
        if name == "luoshu":
            b = r_luoshu(k, idx)
        elif name == "modulo":
            b = r_modulo(k, idx)
        elif name == "random":
            b = r_random(k, idx, rng)
        else:
            b = ch.route(k)
        loads[b] += 1
        row2d[r] += 1; col2d[c] += 1
    return imbalance(loads), max(imbalance(row2d), imbalance(col2d))


def main():
    routers = ["luoshu", "modulo", "random", "consistent_hash"]
    modes = ["uniform", "skewed", "sequential_burst"]
    results = {}
    for mode in modes:
        results[mode] = {}
        for n in NS:
            cell = {r: [] for r in routers}
            cell2d = {r: [] for r in routers}
            for seed in SEEDS:
                rng = np.random.default_rng(1000 * seed + n)
                keys = gen_keys(mode, n, rng)
                for r in routers:
                    rrng = np.random.default_rng(7 * seed + 1)
                    imb, imb2d = run_router(r, keys, rrng)
                    cell[r].append(imb); cell2d[r].append(imb2d)
            results[mode][n] = {
                r: {"imbalance_median": round(float(np.median(cell[r])), 4),
                    "imbalance_2d_median": round(float(np.median(cell2d[r])), 4)}
                for r in routers}

    # verdict at the largest N, uniform + 2D
    def med(mode, n, r, key): return results[mode][n][r][key]
    big = NS[-1]
    verdict = {}
    for mode in modes:
        lu = med(mode, big, "luoshu", "imbalance_median")
        beats_all = all(lu < med(mode, big, o, "imbalance_median")
                        for o in routers if o != "luoshu")
        ties_rr = abs(lu - med(mode, big, "modulo", "imbalance_median")) < 0.02
        verdict[mode] = ("luoshu_wins" if beats_all else
                         "ties_round_robin" if ties_rr else "luoshu_loses")
    lu2d = med("uniform", big, "luoshu", "imbalance_2d_median")
    best_other_2d = min(med("uniform", big, o, "imbalance_2d_median")
                        for o in routers if o != "luoshu")
    verdict["2d_balance_uniform"] = ("luoshu_wins_2d" if lu2d < best_other_2d - 0.01
                                     else "no_2d_advantage")

    out = {"results": results, "verdict": verdict, "magic_square": MAGIC}
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()[:16]

    print("=== Luoshu routing benchmark (N=9000, median imbalance, 1.0=perfect) ===")
    print(f"{'mode':<18}{'luoshu':>9}{'round-robin':>13}{'random':>9}{'cons-hash':>11}")
    for mode in modes:
        row = results[mode][big]
        print(f"{mode:<18}{row['luoshu']['imbalance_median']:>9}"
              f"{row['modulo']['imbalance_median']:>13}"
              f"{row['random']['imbalance_median']:>9}"
              f"{row['consistent_hash']['imbalance_median']:>11}")
    print(f"\n2D balance (uniform): luoshu={lu2d}  best-other={best_other_2d:.4f}")
    print(f"\nVERDICT: {json.dumps(verdict, indent=2)}")
    print(f"results_sha={sha}")


if __name__ == "__main__":
    main()
