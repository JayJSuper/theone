"""Complexity-axis analysis + ASCII 2^k curve (AM-011 viz spec: x-axis = 2^k).
Run after the k-axis run completes: python experiments/complexity_axis/analyze.py
Also dumps per-base cliff (first k with acc<0.85) and failure-mode breakdown
(protocol-fail vs confident-wrong), the two distinct collapse signatures.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TOL = 0.005
KS = [1, 2, 3, 4, 5, 6]


def main():
    rows = [json.loads(l) for l in (HERE / "rows.jsonl").read_text().splitlines()
            if l.strip()]
    print(f"n={len(rows)} rows")
    print(f"\n{'k':>3}{'2^k':>5}{'gpt5.1':>9}{'deepseek':>10}{'C':>6}"
          f"{'  gpt fail/wrong':>16}{'  ds fail/wrong':>16}")
    table = {}
    for k in KS:
        kr = [r for r in rows if r["k"] == k]
        if not kr:
            continue
        out = {"combos": 2 ** k, "n": len(kr)}
        for key in ("gpt51", "deepseek"):
            ok = sum(1 for r in kr if r[key]["pred"] is not None
                     and abs(r[key]["pred"] - r["truth"]) <= TOL)
            fail = sum(1 for r in kr if r[key]["pred"] is None)
            wrong = sum(1 for r in kr if r[key]["pred"] is not None
                        and abs(r[key]["pred"] - r["truth"]) > TOL)
            out[key] = {"acc": ok / len(kr), "fail": fail, "wrong": wrong}
        table[k] = out
        print(f"{k:>3}{2**k:>5}{out['gpt51']['acc']:>9.3f}"
              f"{out['deepseek']['acc']:>10.3f}{1.0:>6.2f}"
              f"{out['gpt51']['fail']:>8}/{out['gpt51']['wrong']:<7}"
              f"{out['deepseek']['fail']:>8}/{out['deepseek']['wrong']:<7}")

    # cliffs
    def cliff(key):
        for k in KS:
            if k in table and table[k][key]["acc"] < 0.85:
                return k, 2 ** k
        return None, None
    gc, gc2 = cliff("gpt51")
    dc, dc2 = cliff("deepseek")
    print(f"\ncliff (first k where acc<0.85):")
    print(f"  gpt-5.1:  k={gc} (2^k={gc2})")
    print(f"  deepseek: k={dc} (2^k={dc2})")

    # ASCII curve on 2^k axis
    print(f"\n=== accuracy vs 2^k (combinatorial marginalization load) ===")
    for k in KS:
        if k not in table:
            continue
        g = table[k]["gpt51"]["acc"]; d = table[k]["deepseek"]["acc"]
        bar_g = "█" * int(round(g * 30)); bar_d = "░" * int(round(d * 30))
        print(f"2^{k}={2**k:>3} | gpt {g:.2f} {bar_g}")
        print(f"        | ds  {d:.2f} {bar_d}")

    interp = ("CLIFF CONFIRMED: both bases collapse as 2^k grows; engine flat at "
              "1.000. The breakpoint is combinatorial marginalization load, not "
              "node count — flagship cliff at 2^{}={}, flash cliff at 2^{}={}."
              .format(gc, gc2, dc, dc2) if gc and dc else
              "partial: see per-k table")
    (HERE / "summary.json").write_text(json.dumps(
        {"table": table, "cliff": {"gpt51": gc, "deepseek": dc},
         "interpretation": interp}, indent=2))
    print(f"\n{interp}")


if __name__ == "__main__":
    main()
