"""Retrieval kernel bake-off (frozen prereg 4bd361a). cosine vs spectral vs HRR
across 3 regimes. Honest verdict per regime. Run: python .../run.py
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from theone.memory.retrieval import (cosine_scores, spectral_scores,
                                     hrr_bind, hrr_unbind)

HERE = Path(__file__).parent
D, N, M, SEEDS = 64, 200, 50, list(range(20))


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def regime_semantic(rng):
    keys = rng.standard_normal((N, D))
    tgt = rng.integers(0, N, M)
    queries = keys[tgt] + 0.3 * rng.standard_normal((M, D))
    return keys, queries, tgt


def regime_shift(rng):
    keys = rng.standard_normal((N, D))
    tgt = rng.integers(0, N, M)
    queries = np.stack([np.roll(keys[t], int(rng.integers(1, D))) for t in tgt])
    return keys, queries, tgt


def regime_bound(rng):
    """Keys = bind(role_j, filler_j). Query = bind(q_role, filler_t): shares the
    target's filler but a different role. Correct retrieval = the key with the
    same filler. HRR unbinds the query's role to expose the filler."""
    roles = [_unit(rng.standard_normal(D)) for _ in range(N)]
    fillers = [_unit(rng.standard_normal(D)) for _ in range(N)]
    keys = np.stack([hrr_bind(roles[j], fillers[j]) for j in range(N)])
    tgt = rng.integers(0, N, M)
    q_roles = [_unit(rng.standard_normal(D)) for _ in range(M)]
    queries = np.stack([hrr_bind(q_roles[i], fillers[tgt[i]]) for i in range(M)])
    return keys, queries, tgt, roles, fillers, q_roles


def topk_hits(scores_matrix, tgt):
    """scores_matrix (M,N): top-1 hit rate."""
    pred = np.argmax(scores_matrix, axis=1)
    return float(np.mean(pred == tgt))


def run_regime(name, rng):
    out = {}
    if name == "bound":
        keys, queries, tgt, roles, fillers, q_roles = regime_bound(rng)
        # cosine / spectral on raw bound vectors
        out["cosine"] = topk_hits(np.stack([cosine_scores(q, keys) for q in queries]), tgt)
        out["spectral"] = topk_hits(np.stack([spectral_scores(q, keys) for q in queries]), tgt)
        # HRR: unbind query by its own role -> recovers filler; compare to each
        # key unbound by ITS role -> recovers each filler; match fillers by cosine
        key_fillers = np.stack([hrr_unbind(keys[j], roles[j]) for j in range(N)])
        rec = np.stack([hrr_unbind(queries[i], q_roles[i]) for i in range(M)])
        out["hrr"] = topk_hits(np.stack([cosine_scores(rec[i], key_fillers)
                                         for i in range(M)]), tgt)
        return out
    keys, queries, tgt = (regime_semantic(rng) if name == "semantic"
                          else regime_shift(rng))
    out["cosine"] = topk_hits(np.stack([cosine_scores(q, keys) for q in queries]), tgt)
    out["spectral"] = topk_hits(np.stack([spectral_scores(q, keys) for q in queries]), tgt)
    out["hrr"] = out["cosine"]                    # HRR == cosine on raw vectors
    return out


def main():
    regimes = ["semantic", "shift", "bound"]
    kernels = ["cosine", "spectral", "hrr"]
    acc = {r: {k: [] for k in kernels} for r in regimes}
    for seed in SEEDS:
        for r in regimes:
            res = run_regime(r, np.random.default_rng(100 * seed + hash(r) % 97))
            for k in kernels:
                acc[r][k].append(res[k])
    table = {r: {k: round(float(np.mean(acc[r][k])), 3) for k in kernels}
             for r in regimes}
    # per-regime winner + paired-better count vs cosine
    verdict = {}
    for r in regimes:
        cos = np.array(acc[r]["cosine"])
        winner = max(kernels, key=lambda k: table[r][k])
        better = {k: int(np.sum(np.array(acc[r][k]) > cos)) for k in kernels if k != "cosine"}
        verdict[r] = {"winner": winner, "acc": table[r], "beats_cosine_count": better}

    out = {"table": table, "verdict": verdict}
    (HERE / "results.json").write_text(json.dumps(out, indent=2))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()[:16]
    print("=== Retrieval bake-off — top-1 hit rate (20 seeds) ===")
    print(f"{'regime':<12}{'cosine':>9}{'spectral':>10}{'hrr':>8}   winner")
    for r in regimes:
        print(f"{r:<12}{table[r]['cosine']:>9}{table[r]['spectral']:>10}"
              f"{table[r]['hrr']:>8}   {verdict[r]['winner']}")
    print(f"\nVERDICT: {json.dumps({r: verdict[r]['winner'] for r in regimes})}")
    dom = all(verdict[r]['winner'] == 'cosine' for r in regimes)
    print("cosine dominates all -> seven-paths retrieval claims FAIL"
          if dom else "complementary: no single kernel wins all -> "
          "supports multi-paradigm-organ thesis")
    print(f"results_sha={sha}")


if __name__ == "__main__":
    main()
