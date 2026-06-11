# CC_TASKBATCH_01 — The One 第一批工程任务令(可原样移交 Claude Code)

**派单人**:第一负责人(Claude)|**执行人**:Claude Code |**预算/密钥**:Jay 唯一持有,任务不得请求密钥
**全局纪律**:禁 TODO/placeholder/mock-only;必须 mock 处写 `# MOCK-SCOPE: <范围> | 转真条件: <条件>`;一切随机显式 seed;实验产物入 `experiments/` 带 SHA256 数据指纹;每任务完成附:变更摘要+测试输出+运行命令实录。
**冻结依据**:T1 数学规格(七断言+A8)、EG 修正案 1/1a/2/2a、A7 终版合取逻辑、δ_min 校准案、两阶段校准通用条款(详见 docs/00_FROZEN_REGISTRY.md,由 CC-13 建册)。

---

## CC-01 仓库骨架与打包
**路径**:仓库根
**产出**:
```
theone/
├── pyproject.toml          # name=theone, Apache-2.0, python>=3.10, deps: numpy, scipy, networkx, pytest, pytest-cov
├── LICENSE                 # Apache-2.0 全文
├── README.md               # 见 CC-12
├── .gitignore
├── .github/workflows/ci.yml
├── docs/00_FROZEN_REGISTRY.md
├── src/theone/__init__.py  # __version__ = "0.1.0"
├── src/theone/{types.py,config.py,cli.py}
├── src/theone/causal/__init__.py
├── src/theone/bench/__init__.py
├── src/theone/memory/__init__.py
├── src/theone/agent/__init__.py
├── src/theone/credential/__init__.py
├── tests/conftest.py
└── experiments/.gitkeep
```
**运行**:`pip install -e ".[dev]" && pytest`(空套件可通过)
**验收**:可编辑安装成功;`python -c "import theone; print(theone.__version__)"` 输出 0.1.0。

## CC-02 核心类型与配置
**路径**:`src/theone/types.py`, `src/theone/config.py`
**接口**:
```python
@dataclass(frozen=True)
class Variable: name: str; states: tuple        # 离散变量
@dataclass
class QueryResult: value: float; method: str; details: dict
@dataclass
class CompareResult: obs_ate: float; int_ate: float; are_different: bool|str; stats: dict
class TheOneConfig: seed:int=42; bootstrap_B:int=1000; alpha:float=0.05; delta_min:float|None=None  # None=未校准,A7 须显式拒跑
```
**测试**:`tests/test_types.py`(构造/不可变/默认值)
**验收**:mypy 通过;delta_min 未设时调用 A7 判定须抛 `CalibrationRequiredError`(防止用未校准阈值出结论)。

## CC-03 因果图
**路径**:`src/theone/causal/graph.py`
**接口**:
```python
class CausalGraph:
    def add_variable(v: Variable); def add_edge(parent:str, child:str)
    def set_cpt(var:str, cpt:dict)      # 键=父状态组合元组, 值=该变量状态分布
    def parents(v)->set; def is_dag()->bool
    def validate()->None                # CPT 完整性+概率和=1(容差1e-9)+无环;违规即抛错
```
构造期断言:任何概率/权重越界 [0,1] 直接拒收(冻结裁定:**校验替代归一化**,禁止 min-max 拉伸)。
**测试**:`tests/test_graph.py`(三节点混杂图构造/环检测/CPT 校验/越界拒收 1e6 用例)
**验收**:全绿;networkx 仅作图算法后端,接口不外泄。

## CC-04 后门识别
**路径**:`src/theone/causal/identify.py`
**接口**:`backdoor_paths(g, X, Y)->list[path]`;`is_blocked(path, Z, g)->bool`(含对撞体开门规则);`find_adjustment_set(g, X, Y)->set|None`(最小后门集,不可识别返回 None)
**测试**:`tests/test_identify.py` 已知图集:三节点混杂(返回{U})、链(空集)、对撞(条件化反开门)、M 型图
**验收**:全部教科书用例通过;None 路径有明确语义(供 Q-C9 审定的"不可说"返回打底)。

## CC-05 查询引擎(F-1 修复本体)
**路径**:`src/theone/causal/engine.py`
**接口**:
```python
class InterventionEngine:
    def __init__(self, graph: CausalGraph)
    def query_observation(self, y, given)->QueryResult   # 沿已识别后门集做后验 P(Z|given) 加权 —— 一般化实现,禁特判
    def query_intervention(self, y, do)->QueryResult     # 切断入边,先验 P(Z) 加权
    def observational_ate(self, X, Y)->float
    def interventional_ate(self, X, Y)->float
    def compare(self, X, Y, config)->CompareResult       # 数值模式(精确推断): 差>容差; 统计模式见 CC-08
```
**冻结语义**(T1):观测查询权重=后验 P(U|X);干预查询权重=先验 P(U)。
**测试**:见 CC-06。
**验收**:CC-06 全绿。

## CC-06 T1 冻结回归测试(sanity gate)
**路径**:`tests/test_confounding_regression.py`
**内容**:按 T1_WORKORDER_F1_FIX.md 原样落地——FROZEN 参数组七断言(A1-A7:0.82/0.70/0.28/0.40/0.54/0.30/True,容差 1e-6)+ 解析真值函数(与实现零共享代码)+ PERTURBED 扰动组 A8 机制守卫 + "守卫的守卫"(解析函数自身复现冻结值)。
**运行**:`pytest tests/test_confounding_regression.py -v`
**验收**:全绿;此文件为 MVP-2A 准入门(sanity gate),CI 标记 required。完成后通知 Jack 复签,F-1 → FIXED,登记册更新。

## CC-07 SCM 生成器
**路径**:`src/theone/bench/scm_generator.py`
**接口**:
```python
@dataclass
class SCMSpec: graph_type:str; confounding_beta:tuple[float,float]; nonlinearity:str; noise:float; n_samples:int; seed:int
class SCMGenerator:
    def generate(spec)->SyntheticSCM     # 含 .sample(n)、.true_int_ate()、.true_obs_ate()(解析或大样本数值真值)
    def grid(grid_config: dict)->Iterator[SCMSpec]   # 网格由配置注入 —— 取值点待 Q-C7 冻结,代码禁止硬编码网格数字
```
混杂强度参数化=标准化 β 乘积(Q-C5 冻结);线性高斯 v0 + 二元离散 v0 两族。
**测试**:`tests/test_scm_generator.py`(seed 复现/真值与手算线性公式对照:β_UX·β_UY=偏差,用 Q-C5 自检 0.6×0.5=0.30 用例)
**验收**:全绿;生成器输出含 spec 指纹。

## CC-08 EG 与 A7 统计判定
**路径**:`src/theone/bench/eg.py`
**接口**:
```python
def eg_score(pred_int, true_int, baseline_int)->float            # 干预误差比
def abs_errors(pred, true)->dict                                  # RMSE/MAE(修正案1:并列主指标)
def a7_judgment(obs_samples, int_samples, config)->CompareResult  # 冻结合取: (1) BCa 95% CI 不含0 ∧ (2)|τ̂|≥δ_min
def bca_ci(stat_fn, data, B, alpha, seed)->tuple                  # BCa bootstrap
def conjunctive_verdict(eg_results, abs_results)->str             # 修正案1a: 同向且均显著=有效;冲突="证据不一致"
```
辅助报告:permutation(强零假设,带声明,不进合取);"统计显著但未达实质阈值"第三出口。**近零效应实例**(|ATE|<ε,ε 入预注册)EG 不归一化改报绝对误差(修正案1)。
**测试**:`tests/test_eg.py`(BCa 对已知分布覆盖率冒烟;A7 用 C6 自检 v2 两情形作固定用例:CI[0.03,0.25]∧0.12≥0.05→True;CI[-0.01,0.25]→False)
**验收**:全绿;δ_min 未校准时 a7_judgment 抛 CalibrationRequiredError。

## CC-09 基准 Runner(MVP-2A)
**路径**:`src/theone/bench/runner.py` + `experiments/mvp2a/`
**功能**:两阶段执行——`--phase calibrate`(校准集:聚合方式 min/product 对照 Spearman 主+Pearson 辅+肯德尔τ tie-break(Q-C4 冻结);δ_min 测绘=max(中位数,0.05);**校准集实例 ID 写入 burned_list.json,正式期加载并拒绝重用**)→ `--phase frozen`(读取冻结配置跑正式网格,产出 EG 全分布+RMSE/MAE+合取判决 JSON+图表)。
**运行**:`theone bench mvp2a --phase calibrate --seed 42` / `--phase frozen --config experiments/mvp2a/frozen.json`
**测试**:`tests/test_runner.py`(burned 隔离强制:正式期注入校准 ID 必须报错)
**验收**:校准→冻结→正式三步在玩具网格(2×2)上端到端跑通;泄漏审计脚本 `experiments/mvp2a/audit_leakage.py` 输出干净。

## CC-10 最小记忆存储
**路径**:`src/theone/memory/store.py`
**接口**:`MemoryStore(path)` — `put(key, value, source:str, ts:float|None)->id` / `get(id)` / `search(key_prefix)` / `delete(id)->bool`(主权删除真删)/ `export()->jsonl`(带走权)。后端 SQLite,真实持久化,非内存 mock。
**测试**:`tests/test_memory.py`(跨进程持久/来源时间戳完整/删除后不可检索/export 完备)
**验收**:全绿;无任何字段缺省吞掉 source。

## CC-11 最小 Agent 编排器
**路径**:`src/theone/agent/orchestrator.py`
**接口**:`Orchestrator(engine, memory)` — `handle(request:str)->AgentResponse(answer, credential:dict)`。v0 为**规则路由**(模式匹配"P(...)"/"do(...)"/记忆指令),`# MOCK-SCOPE: 无 LLM 接入,规则路由 v0 | 转真条件: Demo3 PromptRouter 接入真实基座`。credential v0 字段:query/method/graph_hash/adjustment_set/timestamp/engine_version。
**测试**:`tests/test_agent.py`(三类请求路由正确;凭证字段非空且 graph_hash 可复算)
**验收**:全绿;凭证为真实计算产物,非装饰字符串。

## CC-12 CLI 与 README
**路径**:`src/theone/cli.py`, `README.md`
**命令**:`theone demo causal`(构建 T1 冻结三节点图,打印 0.82/0.70/差异 True 与凭证 JSON)/ `theone test`(转发 pytest)/ `theone bench mvp2a ...`(见 CC-09)
**README**:一句话定义、三行上手、真实状态声明(本版能力=精确推断三节点级离散图+线性 SCM 基准;**不含 LLM、不含反事实、不夸大**)、章程链接、冻结登记册链接。
**验收**:全新环境 `pip install -e . && theone demo causal` 输出冻结真值;README 无一句超出已实现能力。

## CC-13 CI 与冻结登记册
**路径**:`.github/workflows/ci.yml`, `docs/00_FROZEN_REGISTRY.md`
**CI**:push/PR 触发;矩阵 py3.10/3.12;`pytest --cov=theone --cov-fail-under=85`;test_confounding_regression 标记 required;artifact 上传覆盖率。
**登记册初版条目**(一行一资产:编号/一句话定义/冻结日/签字人/原文链接):T1、A7 终版、EG 修正案 1/1a/2/2a、Q-C4/C5 方案、δ_min 校准案、T4 判决门三档+体征 1-2 级、站桩协议 v2、X1 三条件+双锁、两阶段校准通用条款、R3 工序 v1-v3。
**验收**:CI 全绿徽章;登记册渲染正常;此后每次冻结动作由 CC 在同 PR 内更新登记册(流程写入 CONTRIBUTING.md)。

---

## 执行序与汇报
顺序:01→02→03→04→05→06(**到此即请求 Jack 复签,F-1 关闭**)→07→08→09(校准期需等 Q-C7 网格冻结,先以玩具网格打通管线)→10→11→12→13。
每完成一个任务回报:测试输出原文+运行命令实录;任何偏离工单的设计决定,先报协议负责人裁定,不自行变更冻结语义。
