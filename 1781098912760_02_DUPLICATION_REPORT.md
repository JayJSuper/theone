# 02_DUPLICATION_REPORT — 去重分析报告

> 说明：识别 5 个原始文档中**重复出现的结论性章节**，给出保留/归档决策。
> 原文件不动；本表仅指导清洗版主文档（06）应保留哪一份口径。
> 文件代号：F1=全景规划 / F2=审计补充 / F3=Build Phase Gate / F4=Demo-1代码 / F5=Demo-2→v1.0总目录

| Duplicate ID | Section Title | Files Found In | Keep / Archive | Reason |
|---|---|---|---|---|
| D-01 | "最终结论 / 最终审计结论" | F3（≥7 处：6.3、V2、V3 多批）、F2（1 处） | Keep：F2 总体审计结论（保守）；Archive：F3 内 V1/V2/V3 各版本结论 | F3 内多版本互相矛盾，只能留一个保守口径 |
| D-02 | "Gate 通过 / gate_status: PASS" | F3（V2 PASS、V3 PASS ×多次） | Archive 全部"PASS"，改写为 CONDITIONAL_PASS | 与 6.3"有条件通过/43分"冲突，且无真实证据 |
| D-03 | "综合评分" | F3（43/100、88/100、90/100、95/100、98/100） | Keep：43/100（最保守）作为参考；其余 Archive | 同一文件多套互斥评分，保守取低 |
| D-04 | "项目/总进度" | F3(9)、F4(43)、F5(78)、F2(20) | Keep：统一为 04_CANONICAL_STATUS 单一状态表 | 各文件进度口径不一，需收敛为唯一表 |
| D-05 | "本步目标 / 阶段目标" | F4(86)、F5(120)、F3(16)、F2(30) | Keep：F5 路线图 + F4 Demo-1 各一份；其余 Archive | 同义重复，按阶段去重 |
| D-06 | "本步交付清单 / 交付清单" | F5（Demo-2~v1.0 各阶段）、F4（Demo-1） | Keep：F5 各阶段一份；Archive 重复的 Step 小结 | 交付清单按阶段唯一化 |
| D-07 | "验收标准" | F3(27)、F4(39)、F5(70)、F2(44) | Keep：F5 各阶段出口条件 + F2 验收协议；其余 Archive | 收敛为"每阶段一套出口条件" |
| D-08 | "下一步内容 / 后续步骤" | F3、F4、F5 多处 | Keep：06 主文档"下一步工程实现顺序"一份 | 多份下一步互相覆盖，留单一权威 |
| D-09 | Demo-1/2/3/Beta/v1.0 状态表 | F1（路线图状态）、F2（声称vs真实）、F5（能力边界）、F3（Gate） | Keep：F2"声称 vs 真实"为基准；其余作参考 Archive | F2 是唯一区分声称与真实的表，最可信 |
| D-10 | 评分体系（多套） | F3（/100 制）、F2（百分比制）、F4/F5（散落%） | Keep：04 取消打分制，改"状态枚举"；评分仅留 02/03 备查 | 评分制本身互斥，工程上无意义，降级 |
| D-11 | Code Agent 接管说明 | F3（V2 allowed、V3 fully_allowed）、F2、F5 | Keep：`allowed_with_constraints`（保守）；Archive fully_allowed | 无真实证据，不得 fully_allowed |
| D-12 | Release Strategy / 版本状态 | F3（0.1.0~1.0.0 多版本声明）、F5 | Keep：04 release_status 单表（全部降级）；其余 Archive | 多处 released 声明无证据 |
| D-13 | Build Phase Gate V1/V2/V3 | F3（同一文件内三版本并存） | Keep：仅取"事实清单"（ADR/Schema/Event 数量）；Archive 三套结论 | 结论冲突，事实清单可保留为待验证项 |
| D-14 | "The One is for everyone…" 标语 | F3（V2、V3 多次重复） | Keep：1 处置于初心；其余 Archive | 口号重复，保留一次即可 |
| D-15 | "由于篇幅限制/后续实现/实际交付时提供" | F3(8)、F5(4)、F4(3)、F2(1) | Archive 全部 | 占位声明，非真实交付，一律归档 |

## 去重原则小结
- **同一主题多版本结论** → 一律保留最保守的一份，其余归档。
- **打分（/100、%）** → 工程上不作为状态依据，统一改为状态枚举（见 04）。
- **声称 vs 真实** → 以 F2 对照表为唯一基准。
- **占位/篇幅类语句** → 全部归档，不进入干净主文档。
