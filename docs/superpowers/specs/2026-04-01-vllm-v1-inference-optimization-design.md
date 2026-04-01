# Mini vLLM Qwen 推理链路分析与算子优化设计（Triton 导向）

## 1. 背景与目标

### 1.1 背景

当前项目以 `MiniVLLMEngine.generate()` 作为统一推理入口，提供 CLI 和 FastAPI 两种接入形态。核心代码位于：

- `src/core/model.py`
- `src/api/server.py`
- `src/cli/chat.py`

项目已具备：

- 模型加载与生成参数封装（`model_name/max_num_seqs/max_model_len/dtype`）
- API 非流式与“伪流式”输出
- 基础 warmup 与基础指标采集（当前仅请求级 `duration_ms/prompt_len/completion_len`，尚无 TTFT/TPOT/p95/prefill-decode 拆分）

### 1.2 本文目标

1. 从 `generate` 接口出发，映射 vLLM V1 推理主流程（请求接入 -> 调度 -> prefill -> decode -> 采样）。
2. 梳理面向模型推理优化的算子方案，重点给出 Triton 可落地路径。
3. 给出可执行优化蓝图：里程碑、指标、验证与回退策略。

---

## 2. 现状代码链路（从 `generate` 出发）

### 2.1 调用入口

- API 入口：`/v1/chat/completions` 调用 `state.engine.generate(...)`
- CLI 入口：对话循环中调用 `engine.generate(...)`

### 2.2 引擎层

`MiniVLLMEngine.generate()` 执行顺序：

1. `_ensure_loaded()` 懒加载 `vllm.LLM`
2. 读取默认生成参数（`max_tokens/temperature/top_p`）
3. 构造 `SamplingParams`
4. 执行 `self._llm.generate([prompt], sampling_params)`
5. 执行 postprocess 与 follow-up turn 裁剪
6. `stream=False` 返回完整文本；`stream=True` 逐字符 `yield`

### 2.3 关键观察

1. **当前 stream 非 token 级**  
   目前流式分支来自“完整文本后二次逐字符切分”，并非 vLLM 原生 token streaming。

2. **`enforce_eager=True`**  
   会限制编译优化空间，可能影响吞吐与延迟上界。

3. **批量接口仍为串行占位**  
   `generate_batch()` 当前逐条调用，尚未利用 vLLM 批处理优势。

---

## 3. vLLM V1 视角下的推理流程映射

从项目 `generate` 到 vLLM V1 内核路径可抽象为（当前代码对应同步 `LLM.generate()` 路径）：

1. **Request Ingestion**  
   用户请求与采样参数进入引擎。

2. **Scheduler / Continuous Batching**  
   vLLM 在 token 粒度调度多请求，形成动态批。

3. **Prefill 阶段**  
   对完整 prompt 前向，产出首批 logits，并写入 Paged KV Cache。

4. **Decode 阶段**  
   每步仅输入新 token，重用 KV，迭代采样与调度。

5. **Sampling & Stop**  
   基于 `temperature/top_p/stop/max_tokens` 触发停止。

6. **Result Materialization**  
   输出文本回传至项目层，再执行后处理。

---

## 4. 优化目标与约束

### 4.1 已确认目标

- 优化目标：**平衡优化**（TTFT / TPOT / 吞吐综合）
- 改造约束：**中等侵入**
  - 允许新增自定义 Triton 算子
  - 允许替换部分执行路径
  - 不追求一次性大改整个 vLLM 核心

### 4.2 成功标准（建议）

- 基线定义：以 M0 输出的 `benchmark` 数据集为唯一参照，固定 `model_name/dtype/max_model_len/max_num_seqs`、prompt 分桶与并发矩阵。
- TTFT：相对基线下降 >= 15%
- TPOT：相对基线下降 >= 12%
- 吞吐：相对基线提升 >= 15%
- 稳定性：24h 压测无异常退出，错误率无显著上升
- 数值一致性：关键任务集输出质量无显著退化

### 4.3 Triton 集成约束（实现合同）

1. 目标版本：锁定当前仓库依赖的 vLLM 主版本，不在同一阶段跨大版本升级。
2. 集成落点：优先采用“仓库内可维护的算子替换层 + 运行时开关”，避免首次迭代就深度改造 vLLM 调度层。
3. M2 起必须交付“最小可运行集成点”说明：包括模块路径、入口函数、开关变量与 fallback 分支位置。
4. 若某优化必须改动 vLLM 源码，需在该里程碑前补充单独风险评审与回退路径。
5. **M3 decode-attention 执行合同**（目标、输入输出约定、集成边界、回退）见 [`docs/perf/design/decode-attention-contract.md`](../../perf/design/decode-attention-contract.md)；运行时开关为环境变量 **`MINI_VLLM_ENABLE_TRITON_DECODE_ATTN`**（默认关闭，由 `src.core.config.load_runtime_flags()` 读取）。

---

## 5. 算子优化候选与优先级（Triton 倾向）

### 5.1 优先级矩阵

#### P0：Decode Attention（最高优先）

- 原因：通常是 decode 阶段最大热点，直接影响 TPOT 与吞吐。
- Triton 方向：
  - Paged KV 访存友好 kernel
  - 小 batch decode 特化 kernel
- 风险：中（需要严格形状覆盖与一致性验证）

#### P1：RMSNorm/LayerNorm 融合

- 原因：改造成本低，收益稳定。
- Triton 方向：
  - `residual add + rmsnorm` 融合
  - 向量化加载与分块归约
- 风险：低到中

#### P1：MLP 激活链融合

- 原因：prefill 占比较高场景收益明显。
- Triton 方向：
  - `silu/gelu + mul + bias` 等 elementwise 链融合
  - 减少 launch 与中间张量回写
- 风险：中

#### P2：Sampling / Logits 后处理

- 原因：单项收益通常较低，但高频 decode 中可累积。
- Triton 方向：
  - `temperature + topk/topp` 预处理融合
- 风险：中（随机性、可复现与分布一致性验证更复杂）

---

## 6. 可执行优化蓝图（里程碑版）

## M0：基线与可观测性（必须先做）

### 目标

建立可重复、可对比的性能基线，避免“盲调优”。

### 动作

1. 定义固定基准集：
   - Prompt 长度：短/中/长
   - 并发档位：1 / 2 / 4 / 8（按显存可承受上限）
2. 指标分层：
   - 请求层：TTFT、TPOT、tokens/s、p50/p95
   - 阶段层：prefill_ms、decode_ms
   - 算子层：attention/norm/mlp/sampling CUDA time 占比
3. 对比维度：
   - 冷启动 vs 预热后
   - 不同 `max_num_seqs/max_model_len/dtype` 组合
4. 工具链固定：
   - 请求/阶段级：应用层埋点 + 结构化日志
   - 算子级：`nsys profile`（或等价工具）导出 kernel 时间占比报告
5. 结果绑定：
   - 每次压测生成唯一 `run_id`，所有报表与结论必须引用 `run_id`

### 交付物

- `benchmark` 结果表（CSV/Markdown）
- profiling 报告（含热点排序）

---

## M1：流式与观测路径修正

### 目标

将当前“逐字符流式”升级为 token 级可观测路径，为 TPOT 优化提供真实数据。

### 动作

1. 引入 token 级输出路径（对齐 vLLM 能力）
2. API 流式上报 token 时间戳（至少可推导 token 间隔）
3. 指标系统补充：
   - 首 token 到达时间
   - 每 N token 滑动耗时

### 验收

- 线上/测试环境可稳定获得 token 级时序数据
- 与非流式结果一致性通过（同 seed 与同 `SamplingParams` 下，最终完整文本一致；SSE 分块边界允许不同）
- `/v1/chat/completions` 与 `/v1/chat/stream` 的 SSE 帧格式统一

---

## M2：低风险试点（RMSNorm 融合）

### 目标

快速验证 Triton 接入机制与收益评估流程。

### 动作

1. 实现 fused RMSNorm Triton kernel
2. 添加开关与 fallback（环境变量或配置）
3. 在固定形状集合做 AB

### 验收

- 在目标形状上有稳定收益
- 精度/质量验证通过
- fallback 可在异常时即时回退

---

## M3：主收益阶段（Decode Attention Triton 化）

### 目标

命中主热点，显著改善 TPOT 与吞吐。

### 动作

1. 针对 decode 场景的 attention kernel 特化
2. 优先覆盖高频 head_dim / block 配置
3. 按模型配置启用分层策略（先灰度）

### 验收

- TPOT 与吞吐达到阶段目标
- 长上下文 + 多并发稳定性通过
- 错误率与质量不退化

---

## M4：二阶段优化（MLP/采样）

### 目标

吃掉剩余热点，稳步提升整体性能。

### 动作

1. 融合 MLP elementwise 链
2. 按 profiling 决定是否推进 sampling 融合

### 验收

- 当热点占比超过预设阈值（例如 >8%）时推进；否则停止该分支
- 新增开关数量受控（累计不超过 3 个），并保留统一 fallback

---

## 7. 实施细则：验证、回退与风险控制

### 7.1 验证策略

1. **数值验证**
   - 张量级误差阈值：默认 `rtol <= 1e-3`、`atol <= 1e-4`（按 dtype 可调）
   - 覆盖范围：RMSNorm 输出、attention 输出、MLP 输出三个关键节点
2. **任务验证**
   - 固定 prompt 集输出质量对比
3. **性能验证**
   - 与 M0 基线同工况 AB
4. **稳定性验证**
   - 长时间压测 + 多并发压力

### 7.2 回退机制

- 所有 Triton 优化均具备 runtime 开关
- kernel 编译失败、shape 不匹配、异常告警时自动回退默认实现
- 在本仓库采用分环境开关策略（默认关闭 -> 压测环境开启 -> 目标环境启用）

### 7.3 主要风险

1. 形状覆盖不足导致“长尾场景退化”
2. decode 小批次下收益不稳定
3. 质量回归难定位（需强化观测与实验隔离）

---

## 8. 对当前仓库的落地建议（先后顺序）

1. **先做可观测性闭环（M0 + M1）**
   - 没有 token 级时序与阶段拆分数据，不进入 kernel 开发。
2. **再做低风险试点（M2）**
   - 用 RMSNorm 试点验证“开发-验证-回退”闭环。
3. **随后主攻 decode attention（M3）**
   - 这是最可能带来核心收益的阶段。
4. **最后处理次热点（M4）**
   - 以 profiling 结果驱动，不做先验过度优化。

---

## 9. 本设计结论

在“平衡优化 + 中等侵入 + Triton 倾向”的约束下，推荐采用：

- **Profiling 驱动、分阶段替换、可回退发布** 的策略；
- 以 **decode attention** 为核心收益点；
- 以 **RMSNorm 融合** 作为低风险启动点；
- 先建立 token 级观测能力，再推进算子优化。

这条路径能最大化控制工程风险，并确保每一步优化都可量化、可验证、可回退。

## 10. 里程碑追踪表（执行用）

| 里程碑 | 核心产物 | 进入条件 | 完成判定 |
|---|---|---|---|
| M0 | 基线报表 + run_id + profiling 热点表 | 可稳定跑通基准集 | 指标报表可复现 |
| M1 | token 级流式与时序指标 | M0 完成 | 可输出 TTFT/TPOT 与统一 SSE |
| M2 | RMSNorm Triton 试点 + fallback | M1 完成 | 收益与一致性同时达标 |
| M3 | decode attention 优化 | M2 完成 | TPOT/吞吐达到阶段目标 |
| M4 | 次热点优化决策与落地 | M3 完成 | 仅在热点占比达阈值时推进 |
