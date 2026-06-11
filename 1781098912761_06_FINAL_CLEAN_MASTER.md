# 06_FINAL_CLEAN_MASTER — The One 最终干净主文档

> 这是清洗后**唯一**的权威主文档。只有一个状态表、一个 Gate 结论。
> 凡与本文档冲突者，以本文档 + 04_CANONICAL_STATUS 为准。
> 编制于 2026-06-10，基于 5 个原始文档去重、冲突收敛、保守降级而成。

---

## 1. 项目名称
**The One**（The One 1.0）

## 2. 项目初心（不可改写）
> "我希望可以做出一个人人都可以使用，且不归属于任何一家公司的世界通用大模型。这个模型属于 100% 全开源，不融资，不上市，资金需求我完全自己投资覆盖。可以借鉴一切现有大语言模型的能力与优势，以及 DeepSeek 的开源部分，用最佳方式实现，既有现在大模型的能力与优势，又能弥补他们的瓶颈与缺陷。"

口号：**The One is for everyone. By everyone. Forever.**

## 3. 一句话定义
**The One 是一个以因果推理为认知核心、以持续学习为进化机制、以元认知为安全边界、以文化先验为价值锚点、可复用所有开源大模型作为感知层、100% 开源、不属于任何机构的通用智能体。**

## 4. 当前真实状态（唯一口径）
- `project_status: design_to_implementation_transition`（设计到实现的过渡期）
- `gate_status: CONDITIONAL_PASS`
- `code_agent_takeover: allowed_with_constraints`
- `production_readiness: not_yet`
- 核心问题：大量内容仍停留在"设计规格 / Mock"层面，真实实现证据（API/E2E/安全扫描/性能/持续学习）缺失。

## 5. 不能声称的内容
- ❌ 不能说 Demo-2/Demo-3/Beta/v1.0 "已完成"（真实为设计 + Mock / 骨架）
- ❌ 不能说 DeepSeek 已真实集成（仅 Mock，缺真实 API 日志 E-01）
- ❌ 不能说 安全审计/性能基准/GUI 已完成（E-03/E-05/E-06 缺失）
- ❌ 不能说 production-ready / released
- ❌ 不能给出"综合评分 PASS / 88 / 95 / 98"等膨胀结论
- ❌ 不证明 AGI、不证明真正无幻觉、不证明真正内生目标

## 6. 可以声称的内容（有证据）
- ✅ Demo-1 最小认知闭环可工程化运行，单元测试通过
- ✅ 因果图可显式构建；关联查询 P(Y|X) 与干预查询 P(Y|do(X)) 可区分
- ✅ 反事实查询接口可工作
- ✅ 元认知在低置信度时可拒绝编造（"说不知道"）
- ✅ 工作/情景/语义记忆可持久化与检索
- ✅ CLI 可交互
- ✅ Demo-2→v1.0 全阶段路线图与工程蓝图完整、可被 Code Agent 执行

## 7. 五层架构
| 层 | 名称 | 职责 | 关键组件 |
|---|---|---|---|
| L5 | 元认知与内生目标层 | 安全边界、主动拒答 | 七值逻辑、自由能最小化、内生目标生成、φ-最优探索 |
| L4 | 因果推理层（核心） | The One 原生能力 | 因果图、do-calculus、反事实模拟、因果验证 |
| L3 | 记忆与持续学习层 | 跨会话记忆、进化 | 可学习记忆核、工作记忆、情景记忆、语义记忆 |
| L2 | 文化先验层 | 价值锚点 | 64 卦离散空间、内丹火候、儒/道、可配置框架 |
| L1 | 感知与行动层 | 复用开源大模型 | 文本/图像/音频编码（复用 LLM）、工具调用 |

定位：**共生而非替代**——语言/代码/多模态能力复用 DeepSeek/Llama/Qwen/Mistral 等；因果推理、元认知、长期记忆为 The One 原生。

## 8. Demo-1 真实状态 — `implemented_or_near_implemented`
- **In Scope（15 项）**：文本/结构化输入、工作/情景/语义记忆、因果图构建、关联/干预/反事实查询、元认知判断与拒答、文化先验配置、工具调用接口预留、CLI、Demo 脚本。
- **Out of Scope（10 项）**：完整 do-calculus（仅单变量干预）、自动因果发现、真正持续学习、完整内生目标/自由能、完整可学习记忆核、多模态、真实无幻觉、大规模图（≤100 节点/≤500 边）、分布式。
- **真实验证**：types/causal_graph/intervention/counterfactual/working_memory/cli 单元测试通过（如 test_causal_graph 25 passed）。
- **缺口**：真实场景基准未测（见 §18）。

## 9. Demo-2 真实状态 — `design_plus_mock`
- LLMBackend 抽象接口、ModelRegistry 有真实代码、单测通过。
- `deepseek.py` 为**设计 + Mock**，需真实 API Key（缺 E-01）。
- 因果验证管道、元认知过滤为设计规格。

## 10. Demo-3 真实状态 — `partial_design_partial_code`
- `kernel.py`（可学习记忆核）有真实代码、单测通过。
- `ewc.py`（持续学习）、`sleep.py`（睡眠巩固）为**设计规格，功能未验证**（缺 E-04）。

## 11. Beta 真实状态 — `design_specification`
- 安全防护矩阵为设计；`security_audit.py` / `redteam.py` 为**框架代码**。
- 安全扫描（bandit/safety）未真实运行（缺 E-05）；性能基准未真实测量（缺 E-06）。

## 12. v1.0 真实状态 — `future_target`
- `gui/` 骨架代码（HTML/JS 存在，缺完整功能测试 E-03）。
- `api/` 骨架 + 单测，缺端到端测试（E-02）。
- `auth/` 设计规格（单测通过，功能未验证）。

## 13. 最终 canonical status（唯一）
见 `04_CANONICAL_STATUS.md`。要点：
`gate_status: CONDITIONAL_PASS` · `code_agent_takeover: allowed_with_constraints` · `production_readiness: not_yet` · `main_next_step: code_implementation_and_verification`。

## 14. 工程目录结构（Demo-1 冻结版）
```
the_one/
├── README.md  pyproject.toml  requirements.txt  LICENSE(Apache-2.0)
├── docs/        ARCHITECTURE.md  DEMO1_PROTOCOL.md  MODULES.md  ROADMAP.md
├── the_one/
│   ├── __init__.py  __version__.py
│   ├── core/         types.py  config.py  errors.py  logging.py
│   ├── perception/   text_encoder.py  structured_encoder.py
│   ├── memory/       base.py  working_memory.py  episodic_memory.py  semantic_memory.py  memory_store.py
│   ├── causal/       causal_graph.py  intervention.py  counterfactual.py  causal_reasoner.py
│   ├── priors/       cultural_prior.py  default_priors.py
│   ├── metacognition/ uncertainty.py  self_check.py  confidence.py  types.py
│   ├── agent/        the_one_agent.py  response_generator.py
│   └── cli.py
├── tests/        test_memory / test_causal_graph / test_intervention / test_counterfactual / test_metacognition / test_agent / test_demo1
├── examples/     demo1_basic.py  demo1_causal_query.py
└── data/         memory/{working,episodic.db,semantic.json}  configs/cultural_priors.yaml
```

## 15. 核心模块清单
| 层 | 模块 | 真实状态 |
|---|---|---|
| core | types, config, errors, logging | 真实（Demo-1） |
| causal | causal_graph, intervention, counterfactual, causal_reasoner | 真实（Demo-1） |
| memory | working/episodic/semantic, memory_store, kernel | working 真实；kernel 真实；持续学习设计 |
| metacognition | uncertainty, self_check, confidence | 真实（Demo-1） |
| perception | text_encoder, structured_encoder | 真实（Demo-1） |
| agent | the_one_agent, response_generator | 真实（Demo-1） |
| integration | llm_backend, model_registry, backends/deepseek | 前二真实；deepseek Mock |
| learning | ewc, sleep | 设计规格，未验证 |
| api / gui / auth | app, web/, auth | 骨架/设计 |

## 16. 文档清单（清洗后）
00 备份清单 · 01 扫描报告 · 02 去重 · 03 冲突 · 04 唯一状态 · 05 归档映射 · **06 本主文档** · 07 Code Agent 开工包 · 08 Claude Code 实施简报。
工程规格（来自原件，标注"规格存在/效果待验证"）：10 ADR、97 Schema Interface Contract、42 Event Spec、16 Benchmark Roadmap、50 Engineering Story、Domain Model、State Machine、Error Catalog、Config Spec、Security Model、Observability、CI/CD、Dependency Rules。

## 17. 测试清单
- **真实通过（单元）**：types / causal_graph / intervention / counterfactual / memory(working) / metacognition / agent / kernel / llm_backend / model_registry / api(单测) / auth(单测)。
- **Mock**：deepseek 集成测试。
- **缺失**：API E2E、GUI 功能测试、安全扫描真实运行、性能基准真实测量、持续学习长期实验。

## 18. 证据缺口（必须补齐才能升级状态）
| # | 缺口 | 严重度 | 补救 |
|---|---|:---:|---|
| E-01 | DeepSeek 真实 API 调用日志 | 🔴 高 | 获取 API Key 跑真实测试 |
| E-02 | API Server 端到端测试 | 🟠 中 | 编写 E2E |
| E-03 | GUI 完整功能测试 | 🟠 中 | 完成 GUI 实现 |
| E-04 | 持续学习效果验证数据 | 🟠 中 | 运行长期实验 |
| E-05 | 安全扫描真实结果 | 🟠 中 | 安装 bandit/safety 运行 |
| E-06 | 性能基准真实数据 | 🟡 低 | 运行 benchmark.py |

## 19. 下一步工程实现顺序
1. **Phase 1 — 仓库与核心基座**：pyproject.toml → requirements.txt → 包骨架 → core/types → core/errors → core/config → core/logging → core 测试 → CI 基础。
2. **Phase 2 — Demo-1 闭环补全与真实测试**：causal/* 、memory/* 、metacognition/* 、agent/* 、cli，跑通 examples，补真实场景基准（消 E 之外的 Demo-1 缺口）。
3. **Phase 3 — Demo-2 真实集成**：LLMBackend → ModelRegistry → DeepSeek 真实 API（消 E-01）→ 因果验证管道 → 元认知过滤。
4. **Phase 4 — 证据补齐**：API E2E（E-02）、安全扫描（E-05）、性能基准（E-06）。
5. **Phase 5 — Demo-3/Beta/v1.0**：持续学习验证（E-04）、GUI（E-03）、安全/性能/发布。

## 20. Code Agent 接管条件 — `allowed_with_constraints`
- 仅依据本主文档 + 04 + 07 开工，**禁止**参考含冲突的原始稿。
- **不得**新增功能、不得扩写愿景、不得抬升状态、不得把 Mock 写成真实、不得伪造测试结果。
- 每完成一个模块必须有**可运行的真实测试**作为证据，方可在 04 中升级对应状态。
- 触及初心/五层架构/一句话定义需人工确认，不得自行改写。

---
**唯一最终 Gate 结论：`CONDITIONAL_PASS`（不得为好看写 PASS）。**
