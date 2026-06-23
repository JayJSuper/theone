# REPORT — 融合深化③:L3 cognitive_updater(BIC 门控的结构更新凭证)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_cognitive_update/run.py`

## 0. 一句话
当世界漂移、held 因果模型可能失效时,只在**新数据用可复算的 BIC 改善证明**值得更新、且骨架确实改变时,才提议结构更新——否则弃答(保持当前模型、不在噪声上反复 churn)。

## 1. 建了什么(`src/theone/layer3_decision/cognitive_updater.py`)
`CognitiveUpdater`(CredentialedLayer):新数据上重发现候选结构,对 OLD 与候选(同节点集)用 pgmpy BIC 评分。两条弃答:① 候选骨架==当前(无变化)② BIC 改善≤margin(变化不被证明)。否则 ANSWER 提议更新,凭证 value=BIC delta、recompute=确定性重算 delta(gap0)、regime 声明"继承 L2 发现限制(orientation/潜混杂不可证)"。`bic_delta(df, old, new)` 复用。

## 2. 结果(自检 PASS,节点恒 {X,Z,Y})
| 场景 | 行为 |
|---|---|
| A 漂移(Z→Y,X 失效) | **ANSWER**:old [X→Y] → proposed [Z→Y],BIC delta **571.4**(>>margin 10),recompute gap 0 |
| B 平稳(X→Y) | **ABSTAIN**:重发现骨架==当前模型,无结构变化可提议 |

## 3. 含义
- **认知更新也走凭证 + 弃答**:模型修订不是黑箱触发,而是"BIC 可复算证明 + 实际骨架变化"双条件,否则保持当前。
- **继承诚实限制**:提议的新结构仍带 L2 发现腿的 structure-assumed / 潜混杂不可证声明——更新不偷偷"洗白"结构不确定性。
- L3 现在两件事:VFE 决策收敛(decision_layer)+ 认知模型更新(本条),都在脊柱上、都可弃答。

## 4. 下一步
L4 pattern_recognition + conflict_arbitrator(任务#12);L2 潜混杂敏感性界(E-value,任务#13);集成 v2(任务#14)。
