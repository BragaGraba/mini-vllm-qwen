# M4 Go / No-Go（决策门）

本决策对应实现计划 **Task 7**（见 `docs/superpowers/plans/2026-04-01-vllm-inference-optimization-implementation.md` § Task 7），并与设计文档中 **M4：二阶段优化（MLP/采样）** 的验收口径一致（见 `docs/superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md` § M4）。

## 规则

**仅当** profiler 汇总中**剩余主热点占比仍超过 8%**（即 **> 8%**）时，才继续推进 M4 相关优化分支；若热点已不突出（≤ 8%），则**不**再投入该分支，避免无效优化。

Profiler 产物与用法见 [`docs/perf/profiles/README.md`](../profiles/README.md)。
