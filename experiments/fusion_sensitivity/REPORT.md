# REPORT — 融合深化④:潜混杂敏感性界(E-value)上 do() 凭证

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_sensitivity/run.py`

## 0. 一句话
把"latent confounding UNCERTIFIED"从警告变成**可复算的定量界**:每个 do() 现在携带 E-value——一个未建模混杂要多强(风险比尺度、与处理和结果都关联)才能完全解释掉这个效应。

## 1. 建了什么
- `src/theone/layer2_world_model/sensitivity.py`:`e_value_rr`(VanderWeele-Ding,RR≥1: E=RR+√(RR(RR−1)),protective 对称)、`e_value_for_do(p_do1,p_do0)`(do 对比的风险比→E-value + 解释)。
- `CausalLayer` 增量:每个 do() 凭证 evidence 现含 `do_x0` + `sensitivity`(risk_ratio、e_value、interpretation)。**纯增量、行为不变、82 测试零回归**。

## 2. 结果(自检 PASS,X→Y 链)
| 效应 | do(1) | do(0) | RR | E-value |
|---|---|---|---|---|
| 强 p1.8/p0.2 | 0.800 | 0.200 | 4.00 | **7.46**(稳健) |
| 中 p1.65/p0.35 | 0.650 | 0.350 | 1.86 | 3.12 |
| 弱 p1.55/p0.45 | 0.550 | 0.450 | 1.22 | **1.74**(脆弱) |

E-value 随效应大小单调,公式可独立复算。

## 3. 含义
- **诚实的最后一块定量化**:引擎在给定结构下精确,唯一不能认证的是**未建模混杂**(NOTE-004)。E-value 正是这件事的可复算把手——强效应需要"不可能强"的隐混杂才能推翻(稳健),弱效应一个温和隐混杂就可能推翻(脆弱)。
- 与发现腿(NOTE-049)互补:发现腿**声明**潜混杂不可证,敏感性界**量化**它要多强才致命。两者合起来 = 系统对自己最深盲区的诚实 + 定量交代。
- 呼应探针5"variance-bounded, bias-partially-certified":现在 bias 那部分也有了可复算的强度界。

## 4. 下一步
L4 pattern_recognition + conflict_arbitrator(任务#12);集成 v2(任务#14)。
