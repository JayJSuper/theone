# PREREG · 任务二 · EG 推出窄域(强基线 + 非线性)

**冻结时刻**: 本文件 SHA256 在结果落地前记录于 run.py 顶部常量;跑完不改判据。

## 动机
现有 EG=31.1 的诚实边界 = 线性高斯 + 可观测混杂 + **对手是未调整 OLS**(挨打的弱基线)。
本预注册换强基线 + 加非线性,看 EG 真实剩多少。**EG 下降也如实报——这才是真净增益。**

## 数据族(二元 X、二元混杂 U;Y 连续)
混杂结构: U → X, U → Y, X → Y。U 可观测。每族 n=2000/实例,grid 见下。
- **L 线性可加**:  Y = bx·X + bu·U + ε
- **I 交互**:      Y = bx·X + bu·U + bxu·(X·U) + ε
- **N 非线性**:    Y = sigmoid(3·(bx·X + bu·U + bxu·X·U − 0.5)) + ε   (真非线性结构方程)
grid: bx ∈ {0, 0.3, 0.6};bu ∈ {0.5, 1.0};bxu ∈ {0.4, 0.8};噪声 σ ∈ {0.1, 0.3};
每格 30 实例 × 3 族 = 冻结后记总数。

## 真值(do)
ATE = E[Y|do(X=1)] − E[Y|do(X=0)],由结构方程对 U 边缘化**解析/大样本精确**算出(truth)。

## 估计器(同题对照,全部从有限样本估)
- **method(分层 g-computation)**: 按 U 分层,估 E[Y|X=1,U]−E[Y|X=0,U],对 P(U) 平均。
  二元协变量下非参数、对非线性/交互**无偏**。
- **baseline-0 弱(参照)**: 未调整 OLS,Y~X 的 X 系数(原 EG 的对手)。
- **baseline-1 强**: 协变量调整线性回归 Y~X+U 的 X 系数(合格统计学家的默认)。
- **baseline-2 更强**: 协变量调整+交互 Y~X+U+X:U 的 do 估计(把交互项也给足)。

## EG 与判读(冻结)
EG = baseline_err / method_err(>1 ⇒ method 更优),分 (族 × 基线) 报中位数+IQR。
预声明判读:
- 对 **baseline-1(强)** 在 **L 线性**族:预期 EG ≈ 1(调整 vs 调整,method 无独占优势)——如实报。
- 在 **I/N(交互/非线性)**族:误设的线性基线偏,method 胜 ⇒ EG > 1。
- 对 **baseline-2(更强,含交互)**:连交互都给足后,EG 再塌向 1 的程度 = method 的真实剩余优势。
**结论指标 = EG 在 (强基线 baseline-1/2, 非线性 N 族) 下还剩多少。** 接近 1 就说接近 1。

## 可复现命令
```
python experiments/eg_strong_baseline/run.py
```
结果: results.json(各族各基线 EG 中位数+IQR + method/baseline RMSE)+ SHA256SUMS。
