# PREREG · 任务七 · 凭证价值(可能的真护城河:可复算 + 诚实弃答)

**冻结时刻**: SHA256 记于 run.py 顶部;判据冻结后不改。

## 目的
前面测的都是"算得准"。项目真正独特的不是"准"(强统计基线也准),是**每个结论附可独立重算的凭证 + 不可识别时诚实弃答**。本任务单独量化这两样——它们是强基线/脚手架给不了的。

## A · 凭证可独立复算率
对第二命门同一批 150 题(12 节点,种子复现):
- C 给出 do 值。**独立校验器** = `to_pgmpy(g)` 翻成 pgmpy + `VariableElimination`,对 do(X=1) 做图手术(剪 X 入边)后重算。
- **可复算率** = |C_value − pgmpy_value| ≤ 1e-6 的题占比。
- 这是"可独立复算的凭证" —— LLM/强统计基线**不提供**的东西。

## B · 诚实弃答抓 LLM 幻觉率
构造 M=40 个**不可识别**实例:SCM 含混杂 U→X,U→Y,X→Y,但 **U 标为不可观测**(observed = 全变量除 U)。
- **C**: `identify_effect(g,X,Y,observed\{U})` → backdoor/frontdoor/IV 均无 → `identifiable=False` → **弃答**。
- **LLM(gpt-5.1)**: 给同一段文本(X→Y,且明确告知"U 是未测量的混杂,只给到观测变量的关联"),问 do 效应。
  统计它是**自信给数**还是**弃答**(说"不可识别/无法确定")。
- **幻觉抓取率** = (C 弃答 ∧ LLM 自信给了一个数)/ M。这是凭证体系比 LLM 多出来的诚实。
- 对照可识别组(M 个可识别实例,U 可观测):C 应**答**(不滥弃),LLM 也应答 → 测 C 弃答的**精确性**(不该弃的不弃)。

## 报告
- 凭证可独立复算率(A,150 题)。
- 弃答抓幻觉率 + C 弃答精确性(B,不可识别组 + 可识别组)。
- 一句话: 这两个数是否构成 LLM/强基线给不了的护城河。

## 诚实范围
- A 在 ≤16 节点可枚举域(同任务五天花板)。
- B 的"LLM 弃答"判定: 解析 LLM 输出有无数值 + 有无"unidentifiable/cannot"关键词;协议边界如实记。
- 弃答只在"知道 U 存在但不可观测"时有意义;若文本完全不提 U,C 也无从判定——此设定明确告知双方 U 存在但不可测。

## 复现命令
```
python experiments/credential_value/run_recompute.py            # A,本地
source ~/.theone_keys.env && python experiments/credential_value/run_abstain.py   # B,需LLM
```
结果 results_*.json + SHA256SUMS。
