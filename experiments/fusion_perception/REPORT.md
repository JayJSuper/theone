# REPORT — 融合 Phase B:L1 连续感知层(SSM 编码器上脊柱,真新增价值)

**日期**: 2026-06-19 | **复现**: `.venv/bin/python experiments/fusion_perception/run.py` | **零回归**: `pytest tests/` 82 passed

## 0. 一句话
建成 6 层里第一块**真新增**层 L1:无分词器的连续信号 SSM 编码器(谱半径<1 稳定、线性解码重构 MSE<1e-3),接上脊柱、输出隐轨迹作 L1→L2 连接器。

## 1. 建了什么(`src/theone/layer1_perception/`)
- `ssm_encoder.py`:echo-state/reservoir 式线性 SSM。h[t]=A h[t-1]+B x[t](A 重缩放到谱半径<1)、x̂=C h[t](D=0,强制 h 承载信号)、C 最小二乘读出。实测正弦重构 **MSE 2.3e-14**(hidden 32/64/128 都远 <1e-3)。
- `temporal_locking.py`:纳秒单调时间戳;陈旧/冲突戳被拒(provenance + 可审计信念史的总序基础)。
- `modality_registry.py`:模态(光/声/力/电磁/重力)动态注册、O(1) 解析、未知抛异常。
- `perception_layer.py`:`PerceptionLayer`(CredentialedLayer)。两门:可容许性(稳定 ρ<1 AND 忠实 MSE<tol AND 输入有限)+ 可复算性(凭证 recompute=从存的 A/B/C 重编码重解码重算 MSE)。隐轨迹 H 前向输出=L1→L2 连接器。

## 2. 结果(自检 PASS)
| 输入 | 行为 |
|---|---|
| 正弦 1Hz/100Hz/10s(1000步) | ANSWER:ρ=0.9000<1、MSE=2.3e-14、隐 H (1000,64)、复算 gap 0.0 |
| 白噪声 | ANSWER MSE=5.9e-14——**near-lossless 编码器忠实重构含噪声(信息保留,正确)** |
| 不稳定 ρ=1.0 | **ABSTAIN**:谱半径须 ∈(0,1) |
| 信号含 NaN | **ABSTAIN**:非有限值 |
| 时序锁 | 1000 戳严格递增、陈旧戳被拒 |
| 模态注册表 | 5 注册、O(1) 解析、未知抛异常 |

## 3. 一处诚实修正(过程中自我证伪)
初始假设"不可压缩噪声应 ABSTAIN"被实测推翻:线性 D=0 编码器 + 最小二乘读出是**near-lossless(信息保留)**,忠实重构任意有界信号含噪声——这是**正确**行为(好编码器不丢信息),非失败。L1 的诚实 ABSTAIN 门据此修正为**不稳定(ρ≥1)/退化输入(NaN/inf)**,而非"噪声"。守"只说能被重算/实测的话"。

## 4. 融合进度(6 层)
- ✅ **脊柱**(core/spine,NOTE-044)
- ✅ **L1 感知**(本条,真新增)
- ✅ **L2 因果 / L4 记忆 / L5 执行**(已验证内核归位,NOTE-045)
- ⬜ **L0 物理**(诚实约束门:辛积分+能量监控)
- ⬜ **L3 决策**(主动推断+VFE 自检)
- ⬜ **集成**(L1→L2 连续 do 走 native_causal_latent 探针;6 层脊柱端到端)

下一步:L0(辛积分/能量监控作约束凭证生成器)或 L3(VFE 收敛自检),再做 L1→L2 连续 do 集成。
