"""One-command verification of the native verifiable engine — the complete-form heart.

Runs every native / verification-architecture experiment and prints a PASS/FAIL dashboard:
the replay-chain + three-zone + auto-chain verification primitives, the integrated native
engine, the SSM perception front-end, and the double-engine product hot-swap. All CPU/MPS,
no API. Green here = the complete-form heart (perceive -> native verifiable cognition ->
product, two hot-swappable engines) is intact.

Usage:  .venv/bin/python scripts/verify_native.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

PARTS = [
    ("验证原语 · Q1 可重导推导链(replay)", "experiments/bline_derivation_chain/run.py"),
    ("验证原语 · Q3 三区制(稳定+可识别+E-value)", "experiments/bline_three_zone/run.py"),
    ("验证原语 · Q4 自动生成纯重算链", "experiments/bline_auto_chain/run.py"),
    ("心脏 · 原生可验证引擎(估计+链+三区制)", "experiments/native_engine_demo/run.py"),
    ("感知 · SSM 连续流 → 原生可验证 do", "experiments/native_perception/run.py"),
    ("连续 · 原生引擎连续结果(IHDP真实基准)", "experiments/native_continuous/run.py"),
    ("完整体环路 · 连续 感知→原生→产品(长序列收紧)", "experiments/native_perception_continuous/run.py"),
    ("产品 · 双引擎热插拔(符号+原生同吻合)", "experiments/double_engine/run.py"),
    ("完整体 · 一体引擎(感知→识别→可验证do→凭证)", "experiments/complete_form_capstone/run.py"),
    ("完整体 · 认知闭环(验证→记忆→签名召回)", "experiments/complete_form_cognitive_loop/run.py"),
    ("完整体 · 全闭环(感知→验证→记忆→召回→行动/弃答)", "experiments/complete_form_full_loop/run.py"),
    ("完整体 · 压力测试(40随机regime完整性红线)", "experiments/complete_form_stress/run.py"),
    ("完整体 · 连续全闭环(连续结果也记忆+召回+行动)", "experiments/complete_form_continuous_loop/run.py"),
]


def main():
    print("=" * 66)
    print("The One · 原生可验证引擎(完整体心脏)总验证")
    print("=" * 66)
    results = {}
    for label, rel in PARTS:
        path = ROOT / rel
        ok = path.exists() and subprocess.run(
            [PY, str(path)], cwd=str(ROOT), capture_output=True).returncode == 0
        results[label] = ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("-" * 66)
    n = sum(results.values()); allok = n == len(PARTS)
    print(f"完整体心脏: {n}/{len(PARTS)} 通过")
    print(f"OVERALL: {'ALL GREEN — 感知→原生可验证认知→产品(双引擎)心脏成型且自洽' if allok else 'FAILURES PRESENT'}")
    print("\n诚实范围:二元 do() 走 1e-6 可重算符号链;连续结果(IHDP 真实基准)走可复现推断+三区制。")
    print("剩余扩展:大规模训练。神经估计的核验是可复现推断+认知状态分级,非 1e-6 符号重算。")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
