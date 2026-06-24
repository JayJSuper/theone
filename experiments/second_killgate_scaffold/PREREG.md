# PREREG · 第二命门 · 三臂对照(认知层 C vs 裸 LLM vs LLM+脚手架)

**冻结时刻**: 本文件 SHA256 在任何结果落地前计算并记录于 run.py 顶部常量。
**判据冻结后不改**;跑完不许回头改判据。若需偏离,另开 amendment 文件、不改本文件。

## 命门定义
第二命门 = **认知层净增益 vs LLM+脚手架** = `acc(C) - acc(scaffold)`。
这是审计标 0/1 的项。报告该数字无论正/零/负,藏一字作废。

## 题集(与 Baseline B 完全同题,种子复现,不重采)
- `gen_dag(12 nodes)`, `rng = numpy.default_rng(20260614 + 12)`, **前 150 draws**。
- 即 baseline_b_v1_scale 的 L 档 / baseline_b_crossbase 的同一批 150 题。
- 每题: 12 节点因果 SCM,问 P(Y=1 | do(X=1)),truth 由 InterventionEngine 精确算(6 位)。

## 三臂
- **C(认知层)**: theone 引擎,确定性,精确 do。每题重算,无 API。
- **裸 LLM**: gpt-5.1(旗舰,**不准用会崩的 flash 档**),raw 条件,复用 baseline_b_crossbase/rows.jsonl 的同 150 题 per-instance 结果(同题同评分,种子一致)。
- **LLM+脚手架**: gpt-5.1 + 下述"两周能搭出来"的强常规增强,**能力上限给足,不准配弱脚手架**。

## 脚手架定义(冻结 · 强基线)
同一裸 gpt-5.1,加三件常规增强(均为业界标准做法):
1. **思维链(CoT)**: 系统提示要求"逐步推理因果结构、识别后门集、再算"。
2. **工具调用(tool use,核心增强)**: 允许模型把题目解析成 DAG+CPT 并**写一段 Python 代码**做后门调整/枚举,由我方在隔离子进程执行(numpy/itertools,无网络,5s 超时,只读)。模型可据代码返回值给最终答案。这是给 LLM **配上精确计算能力**——脚手架的最强形态。
3. **自洽投票(self-consistency)**: 整条管线独立跑 **K=5** 次(temperature 默认/采样),对 5 个数值答案取**中位数**为该题最终答案。
- 预算: 每次采样 8192 completion tokens(含思考+工具往返),≥裸臂 4096 的两倍,不让脚手架吃亏。
- 协议失败(无法解析出数值/超 K 次工具错误)= 该题记错(与 AM-007 一致)。

## 评分(与 Baseline B 同)
- AM-007: `|pred - truth| <= TOL(0.005)` 记对,否则错;协议失败=错;无重试。
- 三臂同题同评分。

## 主结论 + 统计
- 主指标: acc(C), acc(bare), acc(scaffold); **净增益 = acc(C) - acc(scaffold)**。
- **配对显著性**: C vs scaffold 的 per-instance 对错 → **McNemar 检验**(配对二分),报 p 值。
- 同时报 bare vs scaffold(脚手架相对裸臂提升了多少)。

## 判读(冻结)
- 净增益 > 0 且 McNemar p < 0.05 → 认知层在此题域**优于强脚手架**(第二命门正向)。
- 净增益 ≈ 0(不显著)→ **强脚手架追平认知层**,第二命门**失守**,如实发表负结果。
- 净增益 < 0 → 脚手架胜,第二命门**反向**,如实发表。

## 诚实范围(预声明)
- 此题域 = 12 节点合成 SCM 的 do 查询。**不**外推到真实世界因果/非线性/其他任务。
- 脚手架的代码执行给了它精确计算的可能 → 这是**最严苛**的对手;若 C 仍胜,论点强;若追平,诚实承认 C 在"纯算"上无独占优势,价值需落到别处(凭证/弃答/成本)。

## 可复现命令
```
source ~/.theone_keys.env && python experiments/second_killgate_scaffold/run.py
```
结果文件: results.json(三臂分数+净增益+McNemar)、rows.jsonl(per-instance)、
scaffold_raw/(每题每次采样的原始 LLM 响应+生成代码+执行结果)、SHA256SUMS。
