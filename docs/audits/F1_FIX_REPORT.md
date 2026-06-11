# F1_FIX_REPORT — F-1 修复报告(提交守门员 Jack 复签)

**执行方**:第一负责人亲自执行(代行工程编制)|**日期**:2026-06-11
**依据**:T1_WORKORDER_F1_FIX.md(冻结)|**仓库**:theone v0.1.0(全新正本;旧 Demo-1 降为只读参照矿,裁定见立项书)

## 一、修复语义(实现位置)
- `src/theone/causal/engine.py`:观测查询经**精确联合推断条件化**——混杂 U 的权重自动成为后验 P(U|X),即沿后门路径传播(T1 冻结语义);干预查询经**图手术**——切断 do 变量入边并钳制取值,U 保持先验 P(U)。一般化实现,无任何图结构特判(A8 已验证)。
- `src/theone/causal/graph.py`:构造期校验替代归一化(登记册 VAL-1)——越界概率拒收报错,1e6 极端用例在测试中确认被拒。

## 二、七断言+A8 实测输出(pytest 实跑原文)
```
tests/test_confounding_regression.py::TestF1ConfoundingRegression::test_frozen_truth_table PASSED [ 33%]
tests/test_confounding_regression.py::TestF1ConfoundingRegression::test_frozen_matches_analytic PASSED [ 66%]
tests/test_confounding_regression.py::TestF1ConfoundingRegression::test_mechanism_guard_perturbed PASSED [100%]
============================== 3 passed in 0.17s ===============================
```

## 三、断言比对表
| 断言 | 期望(R3 真值表) | 实测 | 判定 |
|---|---|---|---|
| A1 P(Y=1\|X=1) | 0.82 | 0.820000 | PASS |
| A2 P(Y=1\|do(X=1)) | 0.70 | 0.700000 | PASS |
| A3 P(Y=1\|X=0) | 0.28 | 0.280000 | PASS |
| A4 P(Y=1\|do(X=0)) | 0.40 | 0.400000 | PASS |
| A5 观测 ATE | 0.54 | 0.540000 | PASS |
| A6 干预 ATE | 0.30 | 0.300000 | PASS |
| A7 are_different | True | True | PASS |
| A8 扰动组机制守卫 | 解析真值匹配 | PASSED | PASS |

## 四、全量套件与环境
```
24 passed in 0.68s
```
覆盖率 95%(门槛 85%)。CLI 实跑 `theone demo causal` 输出冻结真值与可复算凭证(graph_hash 可复算,adjustment_set=["U"])。卫生检查:全仓无 TODO/placeholder/病原体字样。

## 五、请求
请对照你的 R3 真值表逐项核签。复签通过则:F-1 → FIXED,登记册 T1 条目更新,本测试文件正式升入 MVP-2A sanity gate(CI 已标记 required)。
