# PREREG · 任务六 · 端到端真相(拆掉"完美解析"便宜)

**冻结时刻**: SHA256 记于 run.py 顶部;判据冻结后不改。

## 目的
第二命门 +5.3 吃了"C 拿结构化 gen_dag dict、LLM 从 render_text 文本解析"的便宜。
让 C 也从**同一段文本**端到端解析+计算,报 C 的 acc 掉到多少、净增益变成多少。
**可能掉到接近 0,如实报。**

## 关键事实(只读勘察确认)
C 的现有语言层 `claim_verifier.W2CG` 只解析**单条因果断言**(cause/effect/方向),
**无法**解析整张多节点 SCM。所以"C 走自己的语言层端到端"**用现有代码不可行**——
这本身是诚实结论:+5.3 依赖一个现有管线无法从文本产出的结构化输入。

## 设计(三档,全部端到端从文本)
对第二命门同一批 150 题(12 节点,种子 20260614+12)的 render_text:
1. **C-oracle(基线对照)**: C 拿结构化 dict(= 原 +5.3 的框架),acc=1.0。
2. **C-tmpl-parser**: **新写一个确定性正则 parser** 解析 render_text 的模板格式
   (变量行/"X and Y directly influence Z"/"P(V=1|cond)=p"/问题行)→ 建图 → 引擎算。
   render_text 是模板化的 → parser 应近乎完美 → 量化它实际 acc。
3. **C-perturbed**: 对 render_text 做**保义扰动**(打乱行序 + "directly influence"↔"are parents of"/"cause"
   同义替换 + P 格式 "P(...)=x"↔"prob ... is x")后,用**同一个** tmpl-parser 解析 → acc 掉多少。
   模拟"现有 parser 遇到非精确模板"的脆性 = 真实世界下界的代理。

## 报告
- acc(C-oracle)=1.0(框架基线)· acc(C-tmpl-parser) · acc(C-perturbed)。
- 净增益 vs 最强 LLM 臂(任务一的 scaffold-v2 = 0.947):
  +5.3(oracle) → ?(tmpl) → ?(perturbed)。
- 一句话: 同一输入下,C 的优势从 +5.3 变成多少。

## 诚实范围
- tmpl-parser 是为本模板手写的、**不是真实 NL parser**;perturbed 仅是脆性代理,非真实凌乱文本。
- 真实世界 NL(任意措辞)无 parser → C 无法端到端 → 优势此处实为未测/接近 0。

## 复现命令
```
python experiments/end_to_end_parse/run.py
```
结果 results.json + SHA256SUMS。
