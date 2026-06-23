# REPORT — 融合深化②:L2 因果发现腿(NOTE-004 落地为可运行的诚实门)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_discovery/run.py` | **零回归**: `pytest tests/` 82 passed

## 0. 一句话
把"结构发现"这块最危险的盲区(NOTE-004:结构正确性不可验证)建成**带可运行诚实门**的发现腿:能检的(有限样本不稳定)用 bootstrap 门拦,不能检的(orientation/潜混杂)在凭证里**永远显式声明**。

## 1. 建了什么(`src/theone/layer2_world_model/`)
- `discovery.py`:`discover`(pgmpy HillClimbSearch/BIC,类别化数据,确定性)+ `bootstrap_stability`(B 次重采样重发现,返回 edge_freq / skeleton_freq / **skeleton_agreement**=重采样复现完整骨架的比例)。
- `discovery_layer.py`:`CausalDiscoveryLayer`(CredentialedLayer)。**两道可容许门**:① skeleton_agreement≥0.8(整骨架在重采样下稳定)② 每条发现链 bootstrap freq≥0.8。可复算门=确定性重发现(同数据→同边,gap0)。regime 永远声明:"orientation within Markov-equivalence AND latent confounding UNCERTIFIED"。

## 2. 结果(自检 PASS,真 SCM U→X,U→Y,X→Y)
| 场景 | 行为 |
|---|---|
| A 观测{U,X,Y} n=2000 | **ANSWER**:骨架稳定度全 1.0(正确恢复骨架);orientation 置信≈0.5(Markov 等价、不可定向);regime 声明限制 |
| B 潜{X,Y} n=2000 | **ANSWER**(X-Y 骨架稳定)**但** do on learned=观测 0.730 vs 真 do 0.610(gap 0.12,自信地错)——凭证 regime 显式声明 latent UNCERTIFIED,**这就是警告不可省的证明** |
| C 微 n=30 | **ABSTAIN**:仅 0.64 重采样复现骨架(<0.8),数据不足不发结构 |

## 3. 诚实机制(与探针5 同构)
- bootstrap 骨架稳定性 = truth-free 可靠性指标,catch **有限样本结构噪声**(可检)——正如探针5 的 subset-spread catch proxy-不完整 bias。
- **但不 catch 潜混杂**:潜混杂在观测数据下产生**稳定地错**的结构(case B),发现腿无法从观测数据分辨 → 凭证**永远声明 latent-confounding-uncertified**。这是"sample-stable, latent-confounding-uncertified",NOTE-004 在系统层的诚实落地,呼应探针5"variance-bounded, bias-partially-certified"。
- **绝不发自信错结构**:数据不足→ABSTAIN;数据足→ANSWER 但限制全声明。

## 4. L2 现在三腿齐全
离散精确 do(`CausalLayer`+pgmpy IPRG)+ 连续声明残差(`ContinuousCausalLayer`)+ **结构发现带可靠性门**(`CausalDiscoveryLayer`),共用一条脊柱。下一步深化:L3 cognitive_updater(BIC 门控结构更新凭证)、L4 pattern/conflict。
