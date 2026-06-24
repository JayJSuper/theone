"""第二命门最终分析 · 含公平脚手架 v2。三臂(+v1失败臂如实并列) + 净增益 + McNemar。
净增益 = acc(C) − acc(max(裸, scaffold-v2))。题域=12节点合成SCM do查询(结构化输入对NL推理)。
Run: python experiments/second_killgate_scaffold/analyze_final.py
"""
from __future__ import annotations
import json, hashlib, importlib.util, sys
from pathlib import Path
from math import comb
import numpy as np

HERE = Path(__file__).parent
TOL = 0.005
spec = importlib.util.spec_from_file_location("scale_run", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec)
_a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _a


def gen():
    rng = np.random.default_rng(20260614 + 12); out = {}
    for i in range(150):
        d = scale.gen_dag(12, rng); g = scale.build_graph(d)
        out[i] = round(float(scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value), 6)
    return out


def ok(p, t): return p is not None and abs(float(p) - float(t)) <= TOL


def load(path, key):
    m = {}
    for ln in Path(path).read_text().splitlines():
        r = json.loads(ln); m[r["i"]] = r.get(key)
    return m


def mcnemar(a_ok, b_ok):
    bb = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    cc = sum(1 for x, y in zip(a_ok, b_ok) if (not x) and y)
    p = min(1.0, 2 * sum(comb(bb + cc, k) for k in range(min(bb, cc) + 1)) / (2 ** (bb + cc))) if (bb + cc) else 1.0
    return bb, cc, p


def main():
    truth = gen()
    bare = {r["i"]: r["gpt51"].get("pred") for r in
            (json.loads(l) for l in (HERE.parent / "baseline_b_crossbase" / "rows.jsonl").read_text().splitlines())}
    v1 = load(HERE / "rows.jsonl", "scaffold_pred")
    v2 = load(HERE / "rows_v2.jsonl", "scaffold_v2_pred")
    ids = sorted(truth)

    C_ok = [True for _ in ids]                                  # engine exact == truth (structured input)
    B_ok = [ok(bare.get(i), truth[i]) for i in ids]
    V1_ok = [ok(v1.get(i), truth[i]) for i in ids]
    V2_ok = [ok(v2.get(i), truth[i]) for i in ids]
    accC, accB, accV1, accV2 = np.mean(C_ok), np.mean(B_ok), np.mean(V1_ok), np.mean(V2_ok)

    # strongest LLM arm = per-instance OR of bare and v2 would be cheating (oracle); use the better ARM's accuracy
    if accV2 >= accB:
        strong_ok, strong_name, strong_acc = V2_ok, "scaffold-v2(自洽)", accV2
    else:
        strong_ok, strong_name, strong_acc = B_ok, "裸gpt-5.1", accB
    bb, cc, p = mcnemar(C_ok, strong_ok)

    out = {
        "n": len(ids), "scope": "C/引擎结构化输入精确(假设解析完美);裸/脚手架从NL文本推理;题域=12节点合成SCM do查询,不外推真实世界",
        "prereg_v1_sha": "de7b328dfdc1aa8bd945c709d4001971e3f9442915e1dacde2a308b52e58c618",
        "amendment2_sha": "df4ebce326de07fbdc907b42fa1a164e58c4ed0f98b01e27816757c2ea963d78",
        "acc": {"C_engine": round(float(accC), 4), "bare_gpt51": round(float(accB), 4),
                "scaffold_v1_FAILED_kept_honest": round(float(accV1), 4), "scaffold_v2_fair": round(float(accV2), 4)},
        "strongest_LLM_arm": strong_name, "strongest_LLM_acc": round(float(strong_acc), 4),
        "SECOND_KILLGATE_net_gain_C_minus_strongest": round(float(accC - strong_acc), 4),
        "mcnemar_C_vs_strongest": {"C_right_LLM_wrong": bb, "C_wrong_LLM_right": cc, "p_exact": round(float(p), 8)},
        "note_v1": "v1脚手架(强制代码)伤性能0.288<裸0.904,不合格强基线,如实保留不用于结论",
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (HERE / "results_final.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sha = hashlib.sha256((HERE / "results_final.json").read_bytes()).hexdigest()
    (HERE / "SHA256SUMS").write_text(f"{sha}  results_final.json\n")
    print(f"\nresults_final.json sha256={sha[:16]}…")
    print(f">>> 第二命门数 = acc(C){accC:.3f} − acc({strong_name}){strong_acc:.3f} = "
          f"{accC-strong_acc:+.4f}  (McNemar p={p:.2g}, n={len(ids)})")


if __name__ == "__main__":
    main()
