# 07_CODE_AGENT_START_PACKAGE — Code Agent 开工包

> 目的：让 Code Agent **只看清洗后的文件**即可直接开始 The One 的工程实现。
> 约束级别：`allowed_with_constraints`。

## 1. 只允许参考的文档
- `06_FINAL_CLEAN_MASTER.md`（唯一主文档）
- `04_CANONICAL_STATUS.md`（唯一状态）
- `07_CODE_AGENT_START_PACKAGE.md`（本文件）
- 如需细节规格，可只读引用 `原始文件副本_只读参考/` 中的内容，但**状态/结论一律以 04、06 为准**。

## 2. 禁止参考的旧文档（会引入冲突）
- ❌ 原始 Build Phase Gate 的 V1/V2/V3 结论与评分（PASS / 88 / 95 / 98）
- ❌ 任何"✅ 已完成 / released / production-ready"的旧声称
- ❌ 含"由于篇幅限制 / 后续实现 / 实际交付时提供"的段落
- ❌ 未清洗的审计报告原文结论

## 3. 第一阶段实现目标（Phase 1: Repository + Core Foundation）
搭建可运行、可测试的仓库骨架与 `core` 模块，作为后续一切实现的地基。**不实现业务功能，不集成 LLM。**

## 4. 第一阶段文件列表（实现顺序）
```
1. pyproject.toml
2. requirements.txt
3. package skeleton (the_one/__init__.py, the_one/__version__.py)
4. the_one/core/types.py      # Node, Edge, Query, Response 等核心类型
5. the_one/core/errors.py     # 自定义异常体系
6. the_one/core/config.py     # 配置加载与管理
7. the_one/core/logging.py    # 日志配置
8. tests/test_core/           # 针对上述 core 的单元测试
9. .github/workflows/ci.yml   # CI 基础（lint + test）
```

## 5. 第一阶段依赖
- Python ≥ 3.10
- 运行期最小依赖：pydantic（类型/校验）、pyyaml（配置）；标准库 logging。
- 开发期：pytest、pytest-cov、ruff（或 flake8）、mypy。
- **不引入** LLM SDK、Web 框架、GUI 依赖（留待后续阶段）。

## 6. 第一阶段测试命令
```bash
pip install -e ".[dev]"
ruff check the_one tests        # 或 flake8
mypy the_one/core
pytest tests/test_core -v --cov=the_one/core --cov-report=term-missing
```

## 7. 第一阶段验收标准
- ✅ `pip install -e .` 成功，包可 `import the_one`
- ✅ `the_one.__version__` 可读取
- ✅ core 四模块（types/errors/config/logging）均有单元测试且**全部真实通过**
- ✅ core 覆盖率 ≥ 85%
- ✅ lint 与 mypy(core) 无错误
- ✅ CI 工作流在干净环境中绿灯
- ✅ 完成后在 04 中将 `release_status.0.1.0-alpha` 由 `evidence_required` 升级，并附测试输出作为证据

## 8. 失败回滚规则
- 任一验收项不达标 → **不进入下一文件/阶段**，先修复。
- 不得为通过而 mock 掉真实断言、不得 `xfail`/`skip` 关键测试来"刷绿"。
- 每个文件以**独立提交**落地；失败时回滚到上一个绿灯提交，不在红灯状态上叠加。

## 9. 不允许自行决定的事项（需人工确认）
- 修改五层架构、一句话定义、项目初心
- 变更包名 `the_one` / 顶层目录结构
- 选择与文档不一致的核心依赖或许可证（许可证固定 Apache-2.0）
- 把任一阶段状态从 design/mock 直接标为"完成"而无真实证据

## 10. 不允许新增的功能
- 不新增超出 06 §8 Demo-1 In-Scope 的能力
- 不提前实现 Demo-2+ 的 LLM 集成、GUI、插件、SDK、企业部署
- 不扩写愿景、不添加未在文档中的模块

## 11. 不允许修改的项目核心主题
- 初心：人人可用、不属任何机构、100% 开源、不融资不上市、自筹资金
- 定位：因果推理为核心、复用开源 LLM 作感知层、共生而非替代
- 五层架构与一句话定义保持原样
