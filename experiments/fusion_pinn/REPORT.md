# REPORT — 融合深化⑧:L0 PINN 物理残差约束(numpy 配点法)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_pinn/run.py` | **零回归**: 102 passed

## 0. 一句话
完成 L0 第三个蓝图模块:在系统真遵守 ODE 处,物理残差约束让灵活模型从"外推爆炸"变成"外推准确"——收益**实测**(非假设),诚实定位守 L0 regime 纪律。

## 1. 建了什么(`src/theone/layer0_physics/pinn_constraint.py`,无需 torch)
`PINNConstraint`:多项式基拟合,可加物理残差惩罚(在配点上强制 q''+ω²q=0)。`fit(lam=0)`=纯数据;`fit(lam>0)`=物理约束。`extrapolation_benefit()` 实测对比。

## 2. 结果(自检 PASS,SHO,训练[0,4] 外推[4,7])
| 模型 | 外推 RMSE |
|---|---|
| 纯数据(无物理先验) | **5.1e4**(无约束多项式外推灾难) |
| 物理约束(PINN) | **0.0044** |
| 改善 | **100%**(远超蓝图 50% 目标) |

## 3. 含义
- **物理先验把外推从爆炸救回**:无约束灵活模型过拟合、外推灾难;物理残差把解钉在定律上 → 外推准确。
- **诚实定位(答2)**:仅在系统真遵守该 ODE 处有效,收益在 SHO 上实测、不外推到任意认知——延续 L0"world view 大胆、verification 诚实"的 regime 纪律。
- L0 现有 3/4 蓝图模块(辛积分 + 能量监控 + PINN);env 接口=plumbing 可选。

## 4. 融合总览(更新)
6 层 + 脊柱 + 8 深化腿 + capstone,14 融合实验 + 102 pytest 全绿。
