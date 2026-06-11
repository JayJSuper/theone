# Claude Code Implementation Brief

## Mission
Implement The One from cleaned canonical documents only.

## Allowed Sources
- 06_FINAL_CLEAN_MASTER.md
- 07_CODE_AGENT_START_PACKAGE.md
- 04_CANONICAL_STATUS.md

## Forbidden Sources
- 原始重复文档（未清洗的 5 个 .txt 的结论部分）
- 旧版 Gate 文档（Build Phase Gate V1/V2/V3 的 PASS 与评分结论）
- 未清洗审计报告
- 含有冲突状态的中间稿
- 任何含 "由于篇幅限制 / 后续实现 / 实际交付时提供" 的段落

## First Implementation Target
Build core package skeleton and core module.
顺序：pyproject.toml → requirements.txt → package skeleton → core/types.py → core/errors.py → core/config.py → core/logging.py → tests for core → CI basic.

## Current Canonical Status (do not contradict)
- gate_status: CONDITIONAL_PASS
- code_agent_takeover: allowed_with_constraints
- production_readiness: not_yet
- main_next_step: code_implementation_and_verification

## Hard Rules
- No new features
- No vision expansion
- No status inflation
- No fake test results
- No overwriting user files
- Tests must run (real, passing — no skip/xfail to fake green)
- Do not mark any stage "done" without real code + real test + real run log
- Apache-2.0 license; package name `the_one`; five-layer architecture and one-line definition are immutable
