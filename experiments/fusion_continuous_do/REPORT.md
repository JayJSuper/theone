# REPORT — 融合深化①:L1→L2 连续 do 连接器(探针1-5 promote 成脊柱层)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_continuous_do/run.py`

## 0. 一句话
把 native_causal_latent 探针 1–5 的可验证连续 do 从实验提升为**包内一等模块 + 脊柱层**,补上感知(L1)与因果(L2)之间的承重连接器:连续隐空间 → 去混杂 do() + 声明残差 + 越界弃答。

## 1. 建了什么
- `src/theone/layer2_world_model/continuous_do.py`:do_estimate(代理上学后门调整的 g-formula do)、subset_spread(truth-free bias 指标,探针5 corr+0.97)、recompute_gap(split-half)。
- `src/theone/layer2_world_model/continuous_causal.py`:`ContinuousCausalLayer`(CredentialedLayer)。两门:可容许性(subset_spread<0.02 AND split-half<0.01,否则 ABSTAIN)+ 可复算性(do 确定性重算 gap 0,vs LLM 采样不可复现)。regime="learned-latent de-confounding; variance-bounded, bias-partially-certified"。

## 2. 结果(自检 PASS,真值 do=0.7268 仅评估用)
| 代理质量 | 行为 |
|---|---|
| clean p=8 σ=0.4 | **ANSWER** do=0.7224(真残差 0.0044)、spread 0.0004、复算 gap 0 |
| ok p=8 σ=0.8 | **ANSWER** do=0.7294(真残差 0.0026)、spread 0.0015、复算 gap 0 |
| noisy p=8 σ=1.6 | **ABSTAIN**(split-half 复算 gap 0.010>0.01) |
| sparse p=1 | **ABSTAIN**(单代理:完整性不可查) |

## 3. 含义
- **L1→L2 缝合从口号到运行的脊柱层**:连续隐空间可验证 do 不再是 native_causal_latent 的支线实验,而是 perception↔causal 之间的一层,带答/弃决策。
- **诚实定位不变**:不让 do 精确(测量误差混杂),价值在残差**有界+随证据收敛+可复算**,bias 高就弃答——绝不 confident-narrow-wrong。
- 与离散 `CausalLayer`(精确 CPT + pgmpy IPRG)互补:离散精确腿 + 连续声明残差腿,共用一条脊柱。

## 4. 下一步深化
L2 因果发现腿(NOTEARS + structure-assumed 凭证)、L3 cognitive_updater、L4 pattern/conflict。
