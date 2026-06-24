# PREREG AMENDMENT-2 · 公平强脚手架(纠正 v1 的反生产脚手架)

**背景**: PREREG v1 冻结的脚手架(强制写代码 + 自洽)实测把 gpt-5.1 从裸臂 0.904 打到
**0.288**(`results.json` sha 58301ca0,如实保留)。该脚手架**伤害**性能 → 不是 JK 要求的
"能力上限给足的强脚手架"。本 amendment **不改 v1 已落地的结果**,另开一个公平强脚手架重测。

**冻结时刻**: 本文件 SHA256 在 v2 结果落地前记录于 run_v2.py 顶部;判据冻结后不改。

## 公平强脚手架定义(scaffold-v2)
同一裸 gpt-5.1,加业界标准的**自洽投票**,套在**直接链式推理**上(裸臂已能 0.904 的方式):
- 系统提示: "逐步推理因果结构与干预,最后给 FINAL: <number>"(允许但**不强制**写代码;
  模型可自行选择是否用代码;若写代码我方仍隔离执行并回灌)。
- **自洽**: 独立采样 **K=5** 次,对数值答案取**中位数**为最终答案。
- 预算: 每次 8192 completion tokens。协议失败=该题记错(同 AM-007)。
- 设计保证: 这是 bare + self-consistency,期望 **≥ bare(0.904)**;若仍 < bare,如实报并取 max。

## 同题同评分
- 同一 150 题(种子 20260614+12,前 150),AM-007,TOL=0.005。
- 三臂: C(引擎精确)/ 裸 gpt-5.1(0.904,crossbase)/ scaffold-v2。
- 主结论: **净增益 = acc(C) − acc(max(裸, scaffold-v2))**;McNemar(C vs 最强LLM臂)。

## 判读(冻结)
- 净增益 > 0 且 McNemar p<0.05 → 引擎在 12 节点 do 题域优于最强 LLM 方法。
- 净增益 ≈ 0 → 强 LLM 追平,如实发表。
- 同时报 scaffold-v2 vs 裸(自洽是否真帮上)。

## 诚实范围(预声明,不变)
C/引擎在**结构化输入**上精确(假设解析完美);裸/脚手架从 NL 文本推理。
题域 = 12 节点合成 SCM 的 do 查询,**不**外推真实世界/非线性其他任务。
v1 失败脚手架(0.288)如实并列报告,不删除、不用 v2 掩盖。

## 可复现命令
```
source ~/.theone_keys.env && python experiments/second_killgate_scaffold/run_v2.py
python experiments/second_killgate_scaffold/analyze.py   # 三臂 + 净增益 + McNemar
```
