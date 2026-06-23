# REPORT — 融合 Phase A:凭证脊柱端到端(6 层身体的灵魂,已跑通)

**日期**: 2026-06-19 | **类型**: 用户《融合路线工程方案》落地第一块承重地基 | **复现**: `.venv/bin/python experiments/fusion_spine/run.py`

## 0. 一句话
把"凭证 + 可独立复算 + 越界弃答"确立为**贯穿全 6 层的强制不变量**,并用可运行代码证明它工作——这是 `docs/FUSION_ARCHITECTURE.md` 判断"超级形态的灵魂"的第一个可验证落地。

## 1. 建了什么(Phase A 缝合 · 新增 `src/theone/core/`)
- `core/contracts.py`:6 层共享数据契约(StateVector/Observation/Graph/Action,蓝图 §4),numpy-only 轻依赖,L0 可无 torch 导入。
- `core/exceptions.py`:与已验证 `theone.types` 对齐(`TheOneError`/`GraphValidationError` 单一来源、不重复定义),扩展 L0/L2 物理/无环异常。
- `core/spine.py`:**凭证脊柱**。`Credential`(value+regime+recompute)、`LayerVerdict`(ANSWER/ABSTAIN)、`CredentialedLayer`(每层基类)、`Spine`(拓扑序运行 + 弃答总线)。
- `experiments/fusion_spine/run.py`:一层一个最小但真实的 gate(L0..L5),端到端证明。

## 2. 核心机制:每层两正交门(os_loop_constrained 推广到 6 层总线)
每层强制两道门:
- **可容许性门**(admissibility):蓝图阈值(L0 能量漂移<1e-3、L1 重构 MSE<1e-3、L3 VFE 单调且<阈值、L5 参考监视器黑名单/沙盒路径…)违反 → ABSTAIN。
- **可复算性门**(recomputability):凭证 `value` 被独立 `recompute()` 复现;**ANSWER 若复算不上 → 脊柱自动降级为 ABSTAIN**。

## 3. 结果(自检 PASS)
| 场景 | 行为 |
|---|---|
| A 健康 | SYSTEM ANSWER,6/6 层,最大复算 gap **0.0** |
| B L0 能量漂移 5e-3 | SYSTEM **ABSTAIN @ L0_physics**(可容许性门) |
| C L2 说谎(声称 do=0.30、自复算=0.45) | SYSTEM **ABSTAIN @ L2_causal**(可复算性门,gap 0.15) |

**证明的灵魂**:系统 ANSWER 当且仅当全 6 层都过两道门;任一可容许性失败→该层 ABSTAIN、下游不跑;**自信但不可复算的声明被脊柱自动降级为 ABSTAIN——LLM 的 confident-narrow-wrong 失败模式被结构性封死,而非寄望避免**。直接呼应 native_causal_latent 探针5。

## 4. 三个决策如何落地
- **答1(全景设计+价值密集执行)**:6 层骨架全部到位(panorama),先把"脊柱"这个最高价值的横切不变量做实。
- **答2(L0 大胆世界观+诚实验证)**:L0 凭证 regime 写明"valid where latent state has Hamiltonian structure"——世界观保持开放,验证守诚实约束门。
- **答3(脊柱由我定、系统跑通才是硬道理)**:脊柱确立为全 6 层强制不变量,并**用运行的代码证明**(逻辑成立+工程跑通+自检 PASS),不靠权威/教条。

## 5. 下一步(价值密集顺序)
1. **Phase A 续**:把已验证的 `causal/`(L2)、`memory/`(L4)、`credential/`+`execution/`(L5)实现 `CredentialedLayer` 接口、归位进 6 层布局,回归 <1e-6(验证胜过规格)。
2. **Phase B**:L1 SSM 编码器 + 时序锁 + 模态注册表 + LLMAdapter,用 native_causal_latent 探针 1–5 作 L1→L2 连接器。
3. **Phase C/D**:L0 辛积分/能量监控(诚实门)、L3 主动推断(VFE 自检)。
4. **Phase E**:6 层凭证脊柱端到端集成。
