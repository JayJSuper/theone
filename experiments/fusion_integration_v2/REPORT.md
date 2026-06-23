# REPORT — 融合集成 v2:data→discover→fit→do 全管线(一条脊柱)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_integration_v2/run.py`

## 0. 一句话
把新建的 L2 三腿串成一条从**原始数据到可验证 do()** 的管线:发现稳定骨架 → 专家定向与数据一致性校验 + 拟合 CPT → do() + IPRG + E-value,全程脊柱凭证 + 弃答总线。

## 1. 建了什么
- `src/theone/layer2_world_model/fit.py`:`fit_cpts(df, oriented_edges)`——在(发现+专家定向)结构上 Laplace 平滑 MLE 拟合 CPT → CausalGraph。**诚实点**:发现给骨架,定向需 obs 之外来源(专家/干预),显式传入并声明。
- `fit_layer.py`:`StructureFitLayer`——校验专家定向骨架==数据发现骨架(不符→ABSTAIN),拟合 CPT,凭证 value=图 content_hash、recompute=重拟合(确定性)。
- `experiments/fusion_integration_v2/`:Spine([CausalDiscoveryLayer, StructureFitLayer, CausalLayer])。

## 2. 结果(自检 PASS,真 do=0.610)
| 场景 | 行为 |
|---|---|
| A 健康 n=2000 | **SYSTEM ANSWER** 3 凭证 `L2d_discovery→L2f_structure_fit→L2_causal`,do=0.611≈真,E-value 4.37 |
| B 不足 n=30 | **ABSTAIN**(发现骨架不完整→与定向不符,数据不支持该结构) |
| C 定向≠数据(丢 U→Y) | **ABSTAIN @ L2f**(定向骨架与数据发现骨架矛盾) |

## 3. 含义
- **从"给定结构算 do"到"从数据走到 do"的完整诚实链**:每一步可弃答——数据不足或定向不符,在发出任何 do() 前就被拦。
- 系统 ANSWER = 层叠回执:发现稳定性 + 拟合可复现 + do 可复算(pgmpy)+ E-value 敏感性界。
- **诚实分工**:数据定骨架(可检)、专家/干预定向(声明来源)、引擎算 do(精确)、E-value 量化潜混杂(可复算)。

## 4. 融合深化全部完成
①L1→L2连续do ②L2发现腿 ③L3认知更新 ④敏感性界 ⑤L4记忆腿 ⑥集成v2。6 层 + 脊柱 + 6 项深化,12 个融合实验全绿,82 测试零回归。
