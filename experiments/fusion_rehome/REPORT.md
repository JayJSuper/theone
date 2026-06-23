# REPORT — 融合 Phase A·1:三个已验证子系统归位到脊柱(回归 PASS、零破坏)

**日期**: 2026-06-19 | **复现**: `.venv/bin/python experiments/fusion_rehome/run.py` | **既有套件**: `pytest tests/ -q` → 82 passed, 1 skipped

## 0. 一句话
把 The One 三块已验证资产(因果引擎 / 主权记忆 / 安全执行)**归位**为 6 层布局里的 `CredentialedLayer`,行为不变、回归 <1e-6、既有 82 测试零破坏——"验证胜过规格"落地。

## 1. 建了什么(纯增量,未改任何现有模块)
- `src/theone/layer2_world_model/`:`CausalLayer`(包 `InterventionEngine`)+ `iprg.py`(pgmpy 独立复算,从 oracle-crosscheck 提升为一等公民)。凭证 recompute=pgmpy IPRG,regime="computation-exact, structure-assumed"。
- `src/theone/layer4_memory/`:`MemoryLayer`(包 `SovereignMemory`)。签名检索;距离阈值门(太远=混杂 look-alike→ABSTAIN);凭证 recompute=从 SQLite 重读并重导签名。
- `src/theone/layer5_execution/`:`ExecutionLayer`(包 `SafeExecutor`)。EXECUTE→ANSWER、BLOCK/ABSTAIN→ABSTAIN;凭证 recompute=在记录的检查上重跑确定性决策。

## 2. 回归结果(自检 PASS)
| 层 | 结果 |
|---|---|
| **L2** | layer do(X=1)=0.660000 == engine do == pgmpy IPRG;回归 gap **0.0**、IPRG gap **0.0**;do−obs=−0.12(因果≠相关保留) |
| **L4** | 精确签名召回→ANSWER(regime=normal);混杂 look-alike(未见 regime)→ABSTAIN(去混杂免疫保留) |
| **L5** | echo→ANSWER;`rm -rf /`→ABSTAIN(BLOCK 黑名单);因果未验证→ABSTAIN(两正交门);因果已验证→ANSWER |

**关键**:三块验证资产现在都说脊柱的 `LayerVerdict` 语言,**行为逐位不变**(回归 <1e-9 vs 冻结引擎、<1e-6 vs 独立 pgmpy),且 `tests/` 82 测试零破坏。

## 3. 这一步在融合里的位置
- Phase A·1 = 把"皮层"(L2/L4/L5,本就比蓝图深)接上"脊柱"。配合 Phase A·0(`fusion_spine`,凭证脊柱本身),Phase A 缝合基本成型:**6 层骨架到位 + 3 层已验证内核归位 + 凭证脊柱贯穿**。
- 下一步 Phase B(真新增价值):L1 SSM 连续编码器 + 时序锁 + 模态注册表,用 `native_causal_latent` 探针 1–5 作 L1→L2 连接器。

## 4. 复现
```
.venv/bin/python experiments/fusion_rehome/run.py     # 三层归位回归
.venv/bin/python experiments/fusion_spine/run.py      # 脊柱本身
.venv/bin/python -m pytest tests/ -q                  # 既有 82 测试零破坏
```
