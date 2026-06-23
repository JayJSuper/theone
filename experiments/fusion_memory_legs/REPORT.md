# REPORT — 融合深化⑤:L4 记忆腿(pattern_recognition + conflict_arbitrator)

**日期**: 2026-06-20 | **复现**: `.venv/bin/python experiments/fusion_memory_legs/run.py` | **零回归**: 82 passed

## 0. 一句话
补完 L4:在**去混杂因果签名**(非表面文本)上做模式识别(复现的因果结构)与冲突仲裁(同一因果问题、矛盾效应),仲裁只**提议**不静默改(主权)。

## 1. 建了什么(`src/theone/layer4_memory/`)
- `pattern_recognition.py`:`PatternRecognizer`。`frequent_edges`(复现的 treatment→target)、`frequent_structures`(复现的完整 structure_key),支持度阈值。
- `conflict_arbitrator.py`:`ConflictArbitrator`。按 structure_key 分组 live 记忆,同一问题(treatment→target|adjustment|regime)效应差>tol 即冲突;提议解决(优先新版本;版本并列→人工复核)。**只提议不删除**(用户拥有记忆)。

## 2. 结果(自检 PASS)
- frequent_edges:`[X,Y]` count 3 support 0.6、`[A,B]` count 2 support 0.4。
- conflicts:**1** 个——`X->Y|adj=[U]|regime=normal` 效应 0.66 vs 0.45(spread 0.21)→ "flag for human review (version tie)"。
- **不误报**:stressed regime 的 X→Y(不同问题)未并入;`A→B` 0.50 vs 0.52(<tol)非冲突。

## 3. 含义
- **判定在因果签名上,不在文本**:真矛盾(同问题不同效应)被抓,regime/文本 look-alike 不被误判——延续 memory_causal_cliff / pillar2 的去混杂检索优势到记忆维护。
- **主权**:仲裁提议(keep-newer / flag),从不静默改写——延续主权记忆原则。
- 两独立同版本源分歧 → 诚实地"人工复核"(不假装自动可解)。

## 4. 进度
L4 记忆腿齐全。融合深化已完成 ①L1→L2连续do ②L2发现腿 ③L3认知更新 ④敏感性界 ⑤L4记忆腿。剩集成 v2(任务#14)。
