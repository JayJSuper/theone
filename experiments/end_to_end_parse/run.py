"""任务六 · 端到端真相: 拆掉"完美解析"便宜。
PREREG 冻结哈希: d643f7e881cd3f2c02748cc2d5552c814e189077f8b575d05cc3c446d5713b18
C-oracle(结构化dict,1.0) / C-tmpl-parser(正则解析render_text) / C-perturbed(保义扰动后同parser)。
Run: python experiments/end_to_end_parse/run.py
"""
from __future__ import annotations
import importlib.util, json, os, re, sys, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
PREREG_SHA = "d643f7e881cd3f2c02748cc2d5552c814e189077f8b575d05cc3c446d5713b18"
assert hashlib.sha256((HERE / "PREREG.md").read_bytes()).hexdigest() == PREREG_SHA, "PREREG changed!"
spec = importlib.util.spec_from_file_location("scale", HERE.parent / "baseline_b_v1_scale" / "run.py")
scale = importlib.util.module_from_spec(spec); _a = sys.argv; sys.argv = ["x"]; spec.loader.exec_module(scale); sys.argv = _a
TOL = 0.005


def parse_render_text(txt):
    """deterministic regex parser for the templated render_text -> gen_dag-style dict, or None on failure."""
    try:
        names = re.search(r"binary variables:\s*(.+?)\.", txt).group(1)
        names = [s.strip() for s in names.split(",")]
        parents = {v: [] for v in names}
        for m in re.finditer(r"^(.+?) directly influence (\w+)\.$", txt, re.M):
            parents[m.group(2)] = [p.strip() for p in m.group(1).split(" and ")]
        # CPTs
        cpts = {v: {} for v in names}
        for m in re.finditer(r"^P\((\w+)=1(?:\|([^)]*))?\)=([\d.]+)$", txt, re.M):
            v, cond, p = m.group(1), m.group(2), float(m.group(3))
            if not cond:
                cpts[v][()] = p
            else:
                vals = {}
                for kv in cond.split(","):
                    k, val = kv.split("="); vals[k.strip()] = int(val)
                combo = tuple(vals[par] for par in parents[v])
                cpts[v][combo] = p
        q = re.search(r"P\((\w+)=1\|do\((\w+)=1\)\)", txt)
        y, x = q.group(1), q.group(2)
        # find u (a common parent of x and y) for the dict shape; not needed by engine
        return {"names": names, "parents": parents, "x": x, "y": y,
                "u": (parents[x][0] if parents[x] else names[0]), "cpts": cpts}
    except Exception:
        return None


def c_answer_from_dict(d):
    g = scale.build_graph(d)
    return scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value


def perturb(txt, rng):
    """保义扰动: 行序打乱 + 同义替换(测 parser 脆性,模拟非精确模板)。"""
    txt = txt.replace(" directly influence ", " cause ")          # 同义 -> parser 正则失配
    txt = re.sub(r"P\((\w+)=1\|([^)]*)\)=([\d.]+)", r"prob \1 given \2 is \3", txt)
    txt = re.sub(r"P\((\w+)=1\)=([\d.]+)", r"prob \1 is \2", txt)
    lines = txt.split("\n")
    body = lines[1:-1]; rng.shuffle(body)                         # 打乱中间行
    return "\n".join([lines[0]] + body + [lines[-1]])


def main():
    print(f"PREREG {PREREG_SHA[:12]}…  task6 end-to-end parse")
    rng = np.random.default_rng(20260614 + 12)
    insts = []
    for i in range(150):
        d = scale.gen_dag(12, rng); g = scale.build_graph(d)
        truth = round(float(scale.InterventionEngine(g).query_intervention(d["y"], 1, {d["x"]: 1}).value), 6)
        insts.append((d, scale.render_text(d), truth))

    ok_oracle = ok_tmpl = ok_pert = parse_fail_tmpl = parse_fail_pert = 0
    prng = np.random.default_rng(7)
    for d, txt, truth in insts:
        ok_oracle += abs(c_answer_from_dict(d) - truth) <= TOL          # structured dict (the +5.3 framing)
        pd = parse_render_text(txt)
        if pd is None: parse_fail_tmpl += 1
        else:
            try: ok_tmpl += abs(c_answer_from_dict(pd) - truth) <= TOL
            except Exception: parse_fail_tmpl += 1
        ptxt = perturb(txt, prng); pp = parse_render_text(ptxt)
        if pp is None: parse_fail_pert += 1
        else:
            try: ok_pert += abs(c_answer_from_dict(pp) - truth) <= TOL
            except Exception: parse_fail_pert += 1

    n = len(insts)
    accs = {"C_oracle_structured_dict": round(ok_oracle / n, 4),
            "C_tmpl_parser_same_text": round(ok_tmpl / n, 4),
            "C_perturbed_text": round(ok_pert / n, 4)}
    scaffold_v2 = 0.9467
    gains = {"vs_scaffold_oracle": round(accs["C_oracle_structured_dict"] - scaffold_v2, 4),
             "vs_scaffold_tmpl": round(accs["C_tmpl_parser_same_text"] - scaffold_v2, 4),
             "vs_scaffold_perturbed": round(accs["C_perturbed_text"] - scaffold_v2, 4)}
    out = {"n": n, "prereg_sha": PREREG_SHA, "acc": accs,
           "parse_failures": {"tmpl": parse_fail_tmpl, "perturbed": parse_fail_pert},
           "net_gain_vs_scaffold_v2_0.947": gains,
           "note": "C的现有W2CG层只解析单claim、无整图parser;tmpl-parser是为本模板手写的确定性正则,非真实NL parser;perturbed是脆性代理"}
    print(json.dumps(out, indent=2, ensure_ascii=False))
    (HERE / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    sha = hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest()
    (HERE / "SHA256SUMS").write_text(f"{sha}  results.json\n")
    print(f"\nresults.json sha256={sha[:16]}…")
    print(f">>> 同一输入下净增益: oracle {gains['vs_scaffold_oracle']:+.3f} → 模板解析 {gains['vs_scaffold_tmpl']:+.3f} → 扰动 {gains['vs_scaffold_perturbed']:+.3f}")


if __name__ == "__main__":
    main()
