"""One-command B-line verification — run every B-line minimal kill-gate probe and print a
PASS/FAIL dashboard. All CPU, no GPU, no API.

Each probe is the minimal, falsifiable feasibility test for one phase of the B-line plan
(docs/THE_ONE_BLINE_PLAN.md). Passing at toy scale means the path is worth scaling;
failing tells us exactly where to pivot.

Usage:  .venv/bin/python scripts/verify_bline.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")

PROBES = [
    ("B1 命门:学到的隐空间保持可验证 do", "experiments/native_causal_latent/run_probe6.py"),
    ("B1 深化:多混杂 + 不可观测混杂诚实声明", "experiments/native_causal_latent/run_probe7.py"),
    ("B2 种子:非自回归可验证结构化生成", "experiments/bline_verifiable_gen/run.py"),
    ("B2 进阶:学到的非自回归生成器(命中88%)", "experiments/bline_learned_gen/run.py"),
    ("B2 结构:非自回归生成整个DAG(构造合法+精确可验)", "experiments/bline_b2_structure/run.py"),
    ("B2 非线性:生成非线性结构,do()经独立模拟核验", "experiments/bline_b2_nonlinear/run.py"),
    ("B3 骨架:SSM O(N) 实测 + 重构", "experiments/bline_ssm_scaling/run.py"),
    ("B3 选择性:选择性O(N) SSM长程召回(胜LTI、无注意力O(N²))", "experiments/bline_b3_selective/run.py"),
    ("B4 种子:神经网原生 do 对引擎吻合", "experiments/bline_native_do/run.py"),
    ("B5 种子:原生可验证认知管线(自凭证+引擎审计)", "experiments/bline_native_pipeline/run.py"),
    ("B5 改进:truth-free 自检弃答(代理可靠性,免oracle)", "experiments/bline_self_abstain/run.py"),
    ("B4 深化:从观测做do + 潜混杂可识别性边界", "experiments/bline_native_do_obs/run.py"),
    ("B3↔B1 整合:O(N) SSM 编码器喂可验证 do", "experiments/bline_ssm_causal/run.py"),
    ("B5 深缝:选择性感知(B3)→ 原生可验证do(胜reservoir)", "experiments/bline_b5_selective_causal/run.py"),
    ("B4/B5 识别:从原始数据发现正确后门集→可验证do(辨混杂vs碰撞)", "experiments/bline_native_discover_do/run.py"),
    ("B2 语言:可验证-by-construction生成(句句可复算+幻觉被拒+诚实弃答)", "experiments/bline_b2_language/run.py"),
    ("B2 序列:学到的非AR条件序列生成(REINFORCE,目标条件化已破塌缩)", "experiments/bline_b2_seqgen/run.py"),
    ("B2 语言2:因果掩码graph-to-text(学到风格+构造上不可幻觉,胜自由生成器)", "experiments/bline_b2_graph2text/run.py"),
    ("B2 语言3:Causal-GAN验证器作判别器(学到的流畅句+0幻觉,DeepSeek想法)", "experiments/bline_b2_causal_gan/run.py"),
    ("B2 语言4:富化4事实多子句(方向+量级+混杂+置信,BC+验证门控0幻觉)", "experiments/bline_b2_causal_gan_rich/run.py"),
    ("B2 语言5:自回归transformer(多句法+验证门控0幻觉,B200架构验证)", "experiments/bline_b2_transformer/run.py"),
    ("W2CG桥:验证任意自然语言因果断言(真/幻觉/弃答,红线0误验)", "experiments/bline_w2cg/run.py"),
    ("W2CG真实语言:DeepSeek生成口语测试集87%+红线0误验(数据飞轮)", "experiments/bline_w2cg_real/run.py"),
    ("W2CG学到版:训练抽取器泛化到未见句94%+红线0误验(胜规则)", "experiments/bline_w2cg_learned/run.py"),
    ("W2CG transformer:8×7广schema序列编码器+规则锚定混合(红线结构性0误验)", "experiments/bline_w2cg_transformer/run.py"),
    ("W2CG propose-verify:学到proposer+grounding安全扩VERIFIED覆盖(胜规则,误验更少)", "experiments/bline_w2cg_propose_verify/run.py"),
    ("金融滩头·真实数据:引擎在真实信贷数据出可复算因果估计+E值+诚实三区(不过度声称)", "experiments/finance_beachhead_real/run.py"),
    ("金融完整体闭环:引擎真数据结构→W2CG核验真实信贷NL断言(真/幻觉/弃答,红线0)", "experiments/finance_beachhead_real/claim_check.py"),
    ("W2CG真实人类文本:维基烟草文章核验真因果句(验真30/弃答81/0误警,语言层合成→真实)", "experiments/w2cg_real_text/run.py"),
    ("B2可验证流畅生成:解耦+round-trip门控(5渲染发/4类幻觉全抓,流畅但无幻觉面)", "experiments/bline_b2_verifiable_fluent/run.py"),
    ("B2统一门控:学到proposer做round-trip(覆盖6%→54%,9×)+规则红线backstop(0误faithful)", "experiments/bline_b2_unified_gate/run.py"),
    ("B4最强:结构无关native do(一网读任意未见DAG算do,对枚举引擎MAE玩具尺度,内化do-calculus算法)", "experiments/bline_native_do_varstruct/run.py"),
    ("B5收口:端到端原生认知(raw样本→估计CPT→结构无关native do→引擎审计;估计随n收敛+审计紧)", "experiments/bline_native_e2e/run.py"),
]


def main():
    print("=" * 64)
    print("The One · B 线最小 kill-gate 总验证(通往终极完整体)")
    print("=" * 64)
    import os
    env = {**os.environ, "THEONE_FAST": "1"}        # heavy GPU probes run a quick smoke config
    results = {}
    for label, rel in PROBES:
        path = ROOT / rel
        ok = path.exists() and subprocess.run(
            [PY, str(path)], cwd=str(ROOT), capture_output=True, env=env).returncode == 0
        results[label] = ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    print("-" * 64)
    n = sum(results.values())
    allok = n == len(PROBES)
    print(f"B 线最小命门: {n}/{len(PROBES)} 通过")
    print(f"OVERALL: {'ALL GREEN — B 线论点在玩具尺度全线成立,值得上真数据/GPU 扩展' if allok else 'FAILURES PRESENT — 据失败处转向'}")
    print("\n诚实范围:全部为 CPU 玩具尺度的可证伪可行性探针,非真实尺度。")
    print("两个真实 kill gate(B1/B4)的全尺度成功仍无保证 —— 见 docs/THE_ONE_BLINE_PLAN.md。")
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
