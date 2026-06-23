# REPORT — 融合深化⑦:L1 LLM 适配器("LLM 提议,The One 验证")

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_llm_adapter/run.py` | **零回归**: 101 passed

## 0. 一句话
把 LLM 作为**感知器官**接入:它的自然语言因果声明被解析成结构化候选(**从不信任**),再由引擎精确 do() 验证——正确则佐证、幻觉数则被驳。定位落到实处:LLM 管流畅/覆盖,The One 管可验证因果真值。

## 1. 建了什么(`src/theone/layer1_perception/llm_adapter.py`)
- `LLMAdapter.parse(text)` → `CausalClaim{treatment, target, effect, adjustment_set, confidence}`。正则解析几类模式("effect of X on Y is 0.3"、"X causes/→ Y"、"adjusting for U")。**优雅降级**:解析不出 treatment/target → confidence 0.1、not actionable。
- `verify_against_engine(claim, engine_effect)`:LLM 声明效应 vs 引擎 do() → corroborated/refuted/unverifiable。

## 2. 结果(自检 PASS)
| 输入 | 解析 |
|---|---|
| "effect of X on Y is 0.30, adjusting for U" | X→Y effect 0.3 adj [U] conf **1.0** actionable |
| "X causes Y" | X→Y effect None conf 0.6 actionable |
| "it's complicated..." | None conf **0.1** not actionable |

验证(引擎 ATE=0.30):LLM 说 0.30→**corroborated**(gap 0);LLM 说 0.60→**refuted**(gap 0.3,幻觉数被抓)。

## 3. 含义
- **"LLM 提议,The One 验证"具体化**:LLM 是 perception organ 不是 oracle,声明只在引擎复算佐证时才被信——自信的错数被复算驳回。
- 接通整个定位:LLM(感知/覆盖)+ The One(可验证因果内核)。解析出的候选可进 L4 记忆(签名)、可被 L2 引擎验证,全程脊柱凭证。
- 诚实:这是简单正则解析器(非魔法 NLP),复杂表达会落入低 confidence→不行动,这正是优雅降级。

## 4. 进度
L1 模块齐全(SSM 编码器 + 时序锁 + 模态注册表 + LLM 适配器)。融合深化 ①-⑦ 完成。
