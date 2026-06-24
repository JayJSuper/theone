"""第二命门分析: 三臂打分 + 净增益 + McNemar。读 scaffold rows.jsonl + crossbase 裸臂 + C(精确)。
诚实标记: 若 acc(scaffold) < acc(bare),脚手架反而更差 → 该脚手架不合格强基线,真实对手取 max。

Run: python experiments/second_killgate_scaffold/analyze.py
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
TOL = 0.005


def correct(pred, truth):
    return pred is not None and abs(float(pred) - float(truth)) <= TOL


def main():
    sc = {}
    for ln in (HERE / "rows.jsonl").read_text().splitlines():
        r = json.loads(ln); sc[r["i"]] = r
    bare = {}
    for ln in (HERE.parent / "baseline_b_crossbase" / "rows.jsonl").read_text().splitlines():
        r = json.loads(ln); bare[r["i"]] = r["gpt51"].get("pred")
    ids = sorted(sc)
    n = len(ids)

    C_ok = [correct(sc[i]["C"], sc[i]["truth"]) for i in ids]          # engine: exact on structured input
    S_ok = [correct(sc[i]["scaffold_pred"], sc[i]["truth"]) for i in ids]
    B_ok = [correct(bare.get(i), sc[i]["truth"]) for i in ids]

    accC, accS, accB = np.mean(C_ok), np.mean(S_ok), np.mean(B_ok)

    # McNemar C vs scaffold (paired discordant)
    b = sum(1 for c, s in zip(C_ok, S_ok) if c and not s)             # C right, scaffold wrong
    c = sum(1 for cc, s in zip(C_ok, S_ok) if (not cc) and s)         # C wrong, scaffold right
    from math import comb
    pmc = sum(comb(b + c, k) for k in range(min(b, c) + 1)) / (2 ** (b + c)) * 2 if (b + c) > 0 else 1.0
    pmc = min(pmc, 1.0)

    adversary = max(accS, accB)
    adv_name = "scaffold" if accS >= accB else "bare(脚手架反而更差→取裸臂)"

    out = {
        "n": n, "prereg_sha": "de7b328dfdc1aa8bd945c709d4001971e3f9442915e1dacde2a308b52e58c618",
        "acc": {"C_engine_exact": round(float(accC), 4), "bare_gpt51": round(float(accB), 4),
                "scaffold": round(float(accS), 4)},
        "net_gain": {"C_minus_scaffold": round(float(accC - accS), 4),
                     "C_minus_bare": round(float(accB and accC - accB), 4),
                     "C_minus_strongest_LLM": round(float(accC - adversary), 4),
                     "strongest_LLM_arm": adv_name},
        "mcnemar_C_vs_scaffold": {"b_C_right_S_wrong": b, "c_C_wrong_S_right": c, "p_exact": round(pmc, 6)},
        "scaffold_hurts_vs_bare": bool(accS < accB),
        "scope": "C/engine 在结构化输入上精确(假设解析完美);裸/脚手架从 NL 文本推理。"
                 "题域=12节点合成SCM的do查询,不外推真实世界。",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()
    (HERE / "SHA256SUMS").write_text(f"{sha}  results.json\n")
    print(f"\nresults.json sha256={sha[:16]}…  (n={n})")
    print(f"第二命门数(C − 最强LLM臂 {adv_name}) = {accC - adversary:+.4f}")


if __name__ == "__main__":
    main()
