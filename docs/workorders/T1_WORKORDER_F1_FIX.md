# T1_WORKORDER — F-1 修复执行工单

> 状态:数学规格已冻结(2026-06-11)| 签字:S2 数学守门员 | 终审:第一负责人
> 执行方:Claude Code(用户侧仓库环境)| 验收:外审环境复跑 + 守门员复签
> 本工单为站桩册"呼吸线"第一项。关闭条件见 §5。

---

## 1. 修复范围

**缺陷 F-1(外审报告 11 号,P0/功能级)**:`causal/intervention.py` 中 `query_observation` 未沿后门路径传播混杂,导致教科书混杂图上 P(Y|X) = P(Y|do(X)),`are_different=False`——核心声称"关联与干预可区分"数值上不成立。

**修复语义(冻结,不得偏离)**:

- 观测查询:`P(Y|X=x) = Σ_u P(Y|X=x,U=u) · P(U=u|X=x)` —— 权重为**后验**(贝叶斯更新,沿后门传播)
- 干预查询:`P(Y|do(X=x)) = Σ_u P(Y|X=x,U=u) · P(U=u)` —— 权重为**先验**(切断 U→X)
- 一般化要求:实现必须对任意已识别后门集 Z 做后验加权,不得特判本用例的图结构。

## 2. 冻结真值表(R3 资产,容差 1e-6)

参数:P(U=1)=0.5;P(X=1|U=1)=0.8,P(X=1|U=0)=0.2;
P(Y=1|X,U):(1,1)=0.9,(1,0)=0.5,(0,1)=0.6,(0,0)=0.2

| # | 断言 | 期望值 |
|---|---|---|
| A1 | P(Y=1\|X=1) | 0.82 |
| A2 | P(Y=1\|do(X=1)) | 0.70 |
| A3 | P(Y=1\|X=0) | 0.28 |
| A4 | P(Y=1\|do(X=0)) | 0.40 |
| A5 | 观测 ATE | 0.54 |
| A6 | 干预 ATE | 0.30 |
| A7 | are_different | True |
| A8 | 机制守卫:扰动参数组上解析真值匹配 | 运行时现算 |

## 3. 回归测试代码(按仓库实际 API 调整导入与构图调用,断言逻辑不得改动)

```python
# tests/test_confounding_regression.py
"""F-1 锁定回归:三节点混杂图 U→X, U→Y, X→Y。
真值表来源:守门员 R3 复核卷(2026-06-11),七断言冻结。
规范:每个断言对应 F-1 的一个具体失效面;不得为凑覆盖率添加无关检查。
"""
import pytest

TOL = 1e-6

# ---- 冻结参数(不得修改) ----
FROZEN = dict(
    p_u1=0.5,
    p_x1_u={1: 0.8, 0: 0.2},
    p_y1_xu={(1, 1): 0.9, (1, 0): 0.5, (0, 1): 0.6, (0, 0): 0.2},
)

# ---- A8 机制守卫用扰动参数(与冻结组不同,期望值运行时现算) ----
PERTURBED = dict(
    p_u1=0.3,
    p_x1_u={1: 0.7, 0: 0.1},
    p_y1_xu={(1, 1): 0.85, (1, 0): 0.4, (0, 1): 0.55, (0, 0): 0.15},
)


def analytic_truth(p):
    """解析真值:观测用后验、干预用先验。
    与被测实现零共享代码——守门员 R1 两公式的直接转写。"""
    out = {}
    for x in (1, 0):
        px_u1 = p["p_x1_u"][1] if x == 1 else 1 - p["p_x1_u"][1]
        px_u0 = p["p_x1_u"][0] if x == 1 else 1 - p["p_x1_u"][0]
        p_x = px_u1 * p["p_u1"] + px_u0 * (1 - p["p_u1"])
        post_u1 = px_u1 * p["p_u1"] / p_x                      # 后验 P(U=1|X=x)
        y11, y10 = p["p_y1_xu"][(x, 1)], p["p_y1_xu"][(x, 0)]
        out[f"obs_{x}"] = y11 * post_u1 + y10 * (1 - post_u1)   # 观测:后验加权
        out[f"do_{x}"] = y11 * p["p_u1"] + y10 * (1 - p["p_u1"])  # 干预:先验加权
    out["obs_ate"] = out["obs_1"] - out["obs_0"]
    out["do_ate"] = out["do_1"] - out["do_0"]
    return out


def build_engine(p):
    """按仓库实际 API 构建三节点图并返回查询引擎。
    需提供:query_observation(target,given) / query_intervention(target,do)
           / compare(...)->are_different"""
    raise NotImplementedError("Claude Code:接入 theone.causal 实际 API")


class TestF1ConfoundingRegression:
    def test_frozen_truth_table(self):
        """A1-A7:冻结真值表(0.82/0.70/0.28/0.40/0.54/0.30/True)"""
        eng = build_engine(FROZEN)
        assert eng.query_observation(y=1, given={"X": 1}) == pytest.approx(0.82, abs=TOL)   # A1
        assert eng.query_intervention(y=1, do={"X": 1}) == pytest.approx(0.70, abs=TOL)     # A2
        assert eng.query_observation(y=1, given={"X": 0}) == pytest.approx(0.28, abs=TOL)   # A3
        assert eng.query_intervention(y=1, do={"X": 0}) == pytest.approx(0.40, abs=TOL)     # A4
        assert eng.observational_ate("X", "Y") == pytest.approx(0.54, abs=TOL)              # A5
        assert eng.interventional_ate("X", "Y") == pytest.approx(0.30, abs=TOL)             # A6
        assert eng.compare("X", "Y").are_different is True                                   # A7

    def test_frozen_matches_analytic(self):
        """守卫的守卫:解析函数自身须复现冻结值(防解析函数写错)"""
        t = analytic_truth(FROZEN)
        for key, val in [("obs_1", .82), ("do_1", .70), ("obs_0", .28),
                         ("do_0", .40), ("obs_ate", .54), ("do_ate", .30)]:
            assert t[key] == pytest.approx(val, abs=TOL)

    def test_mechanism_guard_perturbed(self):
        """A8:扰动参数组——锁机制而非数字,封死硬编码通道"""
        t = analytic_truth(PERTURBED)
        eng = build_engine(PERTURBED)
        assert eng.query_observation(y=1, given={"X": 1}) == pytest.approx(t["obs_1"], abs=TOL)
        assert eng.query_intervention(y=1, do={"X": 1}) == pytest.approx(t["do_1"], abs=TOL)
        assert eng.query_observation(y=1, given={"X": 0}) == pytest.approx(t["obs_0"], abs=TOL)
        assert eng.query_intervention(y=1, do={"X": 0}) == pytest.approx(t["do_0"], abs=TOL)
        assert eng.compare("X", "Y").are_different is True
```

## 4. 执行序(Claude Code)

1. 修 `query_observation`:观测查询沿已识别后门集做后验加权(一般化实现,非特判);
2. 落地上述测试文件,接通 `build_engine` 与实际 API;
3. 同步补 `tests/conftest.py`(init_config fixture)——F-1 验证不得依赖污染环境;
4. 干净环境全量复跑:**601 测试无新增失败 + 本文件全绿**;
5. 产出修复报告:diff 摘要、七断言实测输出 vs 期望值比对表、全量测试统计。

## 5. 验收与关闭

- 外审环境复跑通过(建设者不得自审原则:复跑日志独立留档);
- 守门员对照报告复签;
- 04_CANONICAL_STATUS 更新:F-1 → FIXED(附真值表与报告链接),E-07 关闭;
- 本用例自动进入 InterventionBench 的 sanity 套件(MVP-2A 开跑前的零号检查)。

## 6. R3 工序附件登记

本工单的 R3 附件 = 守门员复核卷全文(A1-A7 手算过程)。
工序条目:凡数学签字类交付,签字人附自出数值自检题及手算答案。
命名纪念:让红灯变绿灯的那三个数——0.82 / 0.70 / 0.12。
