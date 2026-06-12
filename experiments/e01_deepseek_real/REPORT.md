# REPORT — E-01 关闭：DeepSeek 真实 API 调用证据

**日期**: 2026-06-12 | **执行**: CC | **资金**: Jay（API 余额）
**E-01 原文**（审计缺失证据清单 🔴 最高级）: "DeepSeek 真实 API 调用日志——获取 API Key 运行真实测试"

## 证据一：首次真实调用（日志归档）
- `first_real_call.json`：HTTP 200，1.48s，模型 `deepseek-v4-flash`，usage 99 tokens（含 33 reasoning tokens）
- 内容（S1 的第一句话）："Correlation means two variables move together, while causation means one directly causes the other—correlation does not imply causation."（项目史注：S1 的第一句话恰是相关≠因果）

## 证据二：真实集成测试入套件
- `src/theone/llm/deepseek.py`：标准库实现，零新依赖；key 仅在调用时读环境变量，**不存实例、不入日志、不入仓库**（单测断言验证）
- `tests/test_llm.py::TestDeepSeekIntegrationReal`：对真实 API 的往返测试，本地实跑 **passed**；无 key 环境（CI）自动 skip
- 实跑原文：`4 passed in 1.50s`（含真实往返）；全套件 `48 passed`

## 真实发现（Mock 永远给不了的）
`deepseek-v4-flash` 为推理型模型：思考 tokens 从 max_tokens 预算中先扣，预算过小则正文为空。已修正调用纪律并写入测试注释。

## 状态变更
E-01：🔴 缺失 → **✅ 已关闭**（真实日志 + 真实集成测试双证据）。
S1 器官正式可用；M1 双系统闭环的第一块真实拼图落位。

## 安全记录
七把 key 存于 `~/.theone_keys.env`（chmod 600，仓库外）；本仓库与提交历史不含任何密钥（已检查）。
