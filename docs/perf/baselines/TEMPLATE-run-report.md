# 实测执行报告（骨架）

> 使用方式：复制本文件为 `docs/perf/baselines/<run-id>-summary.md`（或同目录下你约定的命名），逐节填空。  
> 设计对照：[`docs/superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md`](../superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md) §4.2、§6、§10。

---

## 0. 测试元信息

| 字段 | 填写 |
|------|------|
| 日期 | YYYY-MM-DD |
| 执行人 | |
| 机器 / GPU / 显存 | |
| 驱动 / CUDA | |
| Python | |
| vLLM 版本 | |
| PyTorch 版本 | |
| 代码 commit | |
| **Run ID** | `run-...` |
| 目标阶段 | `M0` / `M1` / `M2 gate` / `其他` |

---

## 1. 固定配置快照

| 变量 | 值 |
|------|-----|
| `MINI_VLLM_MODEL` | |
| `MINI_VLLM_DTYPE` | |
| `MINI_VLLM_MAX_MODEL_LEN` | |
| `MINI_VLLM_MAX_NUM_SEQS` | |
| `MINI_VLLM_STREAM_MODE` | `char` / `token` |
| `MINI_VLLM_ENABLE_TRITON_RMSNORM` | `true` / `false` |
| `MINI_VLLM_ENABLE_TRITON_DECODE_ATTN` | `true` / `false` |
| 计划并发 | `1,2,4,8` |
| 并发 8 是否可测 | 是 / 否；若否，原因： |

---

## 2. M0 基线矩阵

### 2.1 覆盖检查

- [ ] prompt bucket：`short` / `medium` / `long`
- [ ] 并发：`1` / `2` / `4` / `8`（不可调度项已写明原因）
- [ ] `cold` / `warm`
- [ ] `dtype` × `max_model_len` × `max_num_seqs` 矩阵已跑或已标注跳过

### 2.2 执行命令（粘贴）

```bash
# 示例：场景矩阵 dry-run
# python scripts/benchmark_inference.py --dry-run ...

# 实际压测 / 生产负载（按你的方式填写）
```

### 2.3 产物路径

| 产物 | 路径是否存在 |
|------|----------------|
| `scenarios.csv` | |
| `run_metadata.json` | |
| 其它（如 JSONL 日志） | |

---

## 3. M1 流式与时序

### 3.1 SSE

- [ ] `POST /v1/chat/completions` `stream=true` 正常
- [ ] `GET /v1/chat/stream` 正常
- [ ] 两路均为 `data: <json>` + 空行结束帧

### 3.2 事件字段

- [ ] 流式 JSON 含 `token_ts_ms`
- [ ] 时间戳非递减，可计算间隔

### 3.3 一致性

- [ ] 与同参 `stream=false` 的最终文本一致（允许的 SSE 分块差异已记录）

### 3.4 `/metrics/basic` 抽样（一次请求后）

| 字段 | 值 |
|------|-----|
| `avg_ttft_ms` | |
| `avg_tpot_ms` | |
| `avg_first_token_ms` | |
| `avg_token_window_ms` | |
| `p50_duration_ms` | |
| `p95_duration_ms` | |
| `throughput_tokens_per_s` | |

---

## 4. M2 RMSNorm 试点（若本轮包含）

### 4.1 开关

| 场景 | `TRITON_RMSNORM` | 结果 |
|------|------------------|------|
| Baseline | `false` | |
| Pilot | `true` | |

### 4.2 数值（示例表，可增删行）

| shape / dtype | rtol | atol | 通过 |
|---------------|------|------|------|
| | ≤1e-3 | ≤1e-4 | ✅/❌ |

### 4.3 与 M0 同工况对比摘要

（延迟 / 吞吐 / 显存：择要填写）

---

## 5. Profiling 热点（`docs/perf/profiles/<run-id>-hotspot-summary.md` 或本节摘录）

| 类别 | 占比 (%) | 数据来源 |
|------|----------|-----------|
| attention | | nsys / Nsight UI / stub |
| norm | | |
| mlp | | |
| sampling | | |

**备注**（stub 或缺失时说明）：

---

## 6. Gate 判定（对照设计 §4.2）

> 百分比均以 **相对 M0 锚定 run_id** 的同工况对比为准；无基线则填「待补」并标 ❌。

| Gate | 目标 | 实测 | 通过 |
|------|------|------|------|
| TTFT | 相对基线 ≥ 15% 改善 | | ✅/❌ |
| TPOT | 相对基线 ≥ 12% 改善 | | ✅/❌ |
| 吞吐 | 相对基线 ≥ 15% 提升 | | ✅/❌ |
| 稳定性 | 24h 无关键回归 | | ✅/❌ / N/A |
| 优化开关数 | ≤ 3（见 `OPTIMIZATION_FLAG_ENV_KEYS`） | | ✅/❌ |

**综合结论**：`GO M3` / `NO-GO` / `部分通过`

**原因（1–3 条）**：

1.
2.
3.

---

## 7. 复盘附录

**失败或异常 Top 3**：

1.
2.
3.

**当前最大热点（1 行）**：

**若 NO-GO，下一步（最多 3 条，按优先级）**：

1.
2.
3.

---

## 关联链接

- Baseline 目录说明：[`README.md`](README.md)
- Profiling：[`../profiles/README.md`](../profiles/README.md)
- 实现状态索引：设计 spec [§2.4 `#implementation-status`](../superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md#implementation-status)
