"""One-command REAL-SCALE / capstone verification — the companion to verify_bline.py.

verify_bline.py runs the fast toy-scale falsifiable probes (the daily green dashboard). This runs the
heavier real-scale + integration CAPSTONES that establish the system is Ready end-to-end at scale:
the native-do GNN (structure-general, size-extrapolating, batched real-scale), order-free discovery,
long-stream selective perception, and the whole machine composed at real input size. These are slow
(seconds to minutes each, FAST config), so this is a pre-freeze comprehensive check, not routine.

Usage:  .venv/bin/python scripts/verify_scale.py
"""
from __future__ import annotations
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

CAPSTONES = [
    ("B4 结构无关 native do(任意未见DAG,引擎紧)", "experiments/bline_native_do_varstruct/run.py"),
    ("B4 size-invariant GNN 外推(训K4,5→测K6,7)", "experiments/bline_native_do_gnn/run.py"),
    ("B4 外推无悬崖(K=6..10 误差平坦)", "experiments/bline_native_do_gnn_limit/run.py"),
    ("① 批处理GNN真尺度就绪(disjoint-union,外推保持)", "experiments/bline_native_do_gnn_batched/run.py"),
    ("B5 端到端从原始数据(估计→native do→审计)", "experiments/bline_native_e2e/run.py"),
    ("② 无序结构发现(IDA诚实,0误答)", "experiments/bline_order_free_discovery/run.py"),
    ("③ 完整体处理未知结构(发现→可识别do→发言/弃答)", "experiments/bline_complete_unknown_structure/run.py"),
    ("B5 完整体合一(感知→do→发言→弃答)", "experiments/bline_complete_form/run.py"),
    ("B5 系统级红线MC(从不说错+truth-free弃答)", "experiments/bline_complete_form_mc/run.py"),
    ("④ B3选择性感知规模化(长流recover混杂→do)", "experiments/bline_b3_perception_scale/run.py"),
    ("⑤ 整机真尺度(长流感知→native do→发言/弃答)", "experiments/bline_complete_form_scale/run.py"),
]


def main():
    print("=" * 68)
    print("The One · B 线真尺度 / 整机 capstone 总验证(通往科学/系统 100%)")
    print("=" * 68)
    env = {**os.environ, "THEONE_FAST": "1"}
    results = {}
    for label, rel in CAPSTONES:
        t0 = time.time()
        ok = (ROOT / rel).exists() and subprocess.run(
            [PY, str(ROOT / rel)], cwd=str(ROOT), capture_output=True, env=env).returncode == 0
        results[label] = ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}  ({time.time()-t0:.0f}s)")
    print("-" * 68)
    n = sum(results.values())
    allok = n == len(CAPSTONES)
    print(f"真尺度/整机 capstone: {n}/{len(CAPSTONES)} 通过")
    print(f"OVERALL: {'ALL GREEN — 原生可验证认知核在真尺度+整机层面全线成立' if allok else 'FAILURES PRESENT — 据失败处转向'}")
    print("\n诚实范围:FAST 配置(玩具→中尺度);真尺度数值见各 NOTE(云端 256M/200k图等)。")
    print("枚举/精确引擎始终是可复算 oracle;潜混杂受可识别性界;因果序在②③外仍假定已知。")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
