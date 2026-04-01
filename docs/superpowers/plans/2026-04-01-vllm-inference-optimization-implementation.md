# vLLM Inference Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measurable and rollback-safe optimization pipeline for this Mini vLLM project, deliver M0-M2 in code, and complete implementation contracts for M3/M4 handoff with clear AB evidence.

**Architecture:** Keep `MiniVLLMEngine.generate()` as the stable business entrypoint while improving observability, token-level streaming, and kernel backends underneath. Implement in milestones M0->M4 with strict baseline `run_id`, feature flags, and fallback-first strategy so each optimization can be enabled/disabled independently.

**Tech Stack:** Python, FastAPI, vLLM, pytest, Triton, nsys profiler, in-repo Markdown/CSV reports.

**Scope Note:** This plan implements M0-M2 and M3/M4 execution contracts only. Actual M3 decode-attention kernel development and M4 MLP/sampling fusion are tracked as follow-on implementation plans after M2 proof gates pass.

**Design spec (kept in sync with implementation):** [`docs/superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md`](../specs/2026-04-01-vllm-v1-inference-optimization-design.md) — see §2.4 *implementation status* for code anchors.

---

## File Structure Mapping (before tasks)

- `src/core/metrics.py`
  - Extend from basic request metrics to token/stage-level metrics model.
- `src/core/model.py`
  - Add optional token-level streaming path and optimization runtime switches.
- `src/api/server.py`
  - Unify SSE framing behavior and emit richer streaming metrics.
- `src/core/config.py`
  - Add env-driven feature flags for streaming mode and Triton optimization toggles.
- `src/core/benchmarking.py` (new)
  - Shared benchmark utilities (`build_run_id`, scenario schema) importable by tests and scripts.
- `tests/test_engine.py`
  - Add unit tests for token-level stream path and fallback behavior.
- `tests/test_api.py`
  - Add SSE contract tests and stream/non-stream consistency tests.
- `scripts/benchmark_inference.py` (new)
  - Reproducible benchmark runner outputting `run_id` + per-scenario records.
- `scripts/profile_inference.sh` (new)
  - Wrapper for `nsys profile` with deterministic output location.
- `docs/perf/baselines/` (new folder)
  - Store baseline CSV/Markdown reports and run metadata.
- `docs/perf/profiles/` (new folder)
  - Store profiling summaries and references to raw profiler artifacts.

## Hard Acceptance Gates (must satisfy before completion)

- Gate 1: TTFT improves >=15% versus M0 baseline (`run_id` anchored).
- Gate 2: TPOT improves >=12% versus M0 baseline (`run_id` anchored).
- Gate 3: Throughput improves >=15% versus M0 baseline (`run_id` anchored).
- Gate 4: 24h stress run shows no critical regression.
- Gate 5: Optimization switch count remains <=3 with unified fallback behavior.

---

### Task 1: Establish M0 Benchmark Baseline Framework

**Files:**
- Create: `src/core/benchmarking.py`
- Create: `scripts/benchmark_inference.py`
- Create: `docs/perf/baselines/README.md`
- Modify: `src/core/config.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_benchmark_run_id_format():
    from src.core.benchmarking import build_run_id
    run_id = build_run_id("qwen", "fp16")
    assert run_id.startswith("run-")
    assert len(run_id) > 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_benchmark_run_id_format -v`  
Expected: FAIL with missing helper/function.

- [ ] **Step 3: Write minimal implementation**

```python
def build_run_id(model: str, dtype: str) -> str:
    ts = int(time.time())
    return f"run-{model}-{dtype}-{ts}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py::test_benchmark_run_id_format -v`  
Expected: PASS

- [ ] **Step 5: Implement benchmark script scenarios**

Run: `python scripts/benchmark_inference.py --help`  
Expected: options for prompt bucket and concurrency matrix exist.

- [ ] **Step 6: Add M0 required dimensions**

Run: `python scripts/benchmark_inference.py --mode cold,warm --dtype fp16 --max-model-len 2048,4096 --max-num-seqs 1,2,4`  
Expected: output includes cold-vs-warm labels and full parameter matrix metadata.

- [ ] **Step 7: Extend matrix to spec-complete sweep**

Run: `python scripts/benchmark_inference.py --prompt-buckets short,medium,long --concurrency 1,2,4,8`  
Expected: benchmark supports 1/2/4/8; if 8 is not feasible on current VRAM, output must explicitly mark it as skipped with reason.

- [ ] **Step 8: Commit**

```bash
git add src/core/benchmarking.py scripts/benchmark_inference.py src/core/config.py tests/test_engine.py docs/perf/baselines/README.md
git commit -m "feat: add reproducible inference benchmark baseline framework"
```

---

### Task 2: Upgrade Metrics Model to Support TTFT/TPOT/Stage Metrics

**Files:**
- Modify: `src/core/metrics.py`
- Modify: `src/api/server.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
def test_metrics_snapshot_includes_ttft_and_tpot():
    data = basic_metrics.snapshot(window_seconds=300)
    assert "avg_ttft_ms" in data
    assert "avg_tpot_ms" in data

def test_metrics_snapshot_includes_percentiles_and_throughput():
    data = basic_metrics.snapshot(window_seconds=300)
    assert "p50_duration_ms" in data
    assert "p95_duration_ms" in data
    assert "throughput_tokens_per_s" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_metrics_snapshot_includes_ttft_and_tpot -v`  
Expected: FAIL due to missing keys.

- [ ] **Step 3: Write minimal implementation**

```python
def record(..., ttft_ms: float | None = None, tpot_ms: float | None = None):
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_metrics_snapshot_includes_ttft_and_tpot -v`  
Expected: PASS

- [ ] **Step 5: Add stage-level placeholders**

Run: `pytest tests/test_api.py::test_metrics_snapshot_includes_stage_keys -v`  
Expected: PASS with `avg_prefill_ms` and `avg_decode_ms` present.

- [ ] **Step 6: Add rolling window token timing fields**

Run: `pytest tests/test_api.py::test_metrics_snapshot_includes_token_timing_windows -v`  
Expected: PASS with fields for first-token and per-N-token windows.

- [ ] **Step 7: Commit**

```bash
git add src/core/metrics.py src/api/server.py tests/test_api.py
git commit -m "feat: extend metrics with ttft tpot and stage-level fields"
```

---

### Task 3: Implement Token-Level Streaming Path (M1)

**Files:**
- Modify: `src/core/model.py`
- Modify: `src/api/server.py`
- Modify: `src/core/config.py`
- Test: `tests/test_engine.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_stream_emits_token_chunks_not_chars():
    chunks = list(engine.generate("hi", stream=True))
    assert len(chunks) < len("".join(chunks))  # coarse check: not 1-char chunks
```

```python
def test_stream_and_non_stream_have_same_final_text():
    non_stream = client.post("/v1/chat/completions", json=payload_non_stream).json()
    stream_text = collect_stream_text(payload_stream)
    assert stream_text == non_stream["choices"][0]["message"]["content"]

def test_stream_events_contain_token_timestamps():
    events = collect_stream_events(payload_stream)
    assert all("token_ts_ms" in e for e in events if e.get("choices"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_engine.py::test_generate_stream_emits_token_chunks_not_chars tests/test_api.py::test_stream_and_non_stream_have_same_final_text -v`  
Expected: FAIL on current char-based streaming.

- [ ] **Step 3: Write minimal implementation**

```python
# Add stream mode switch:
# MINI_VLLM_STREAM_MODE=char|token
# keep char as fallback, add token mode when backend supports it
```

- [ ] **Step 4: Unify SSE frame behavior across endpoints**

Run: `pytest tests/test_api.py -k stream -v`  
Expected: `/v1/chat/completions` and `/v1/chat/stream` use same framing contract.

- [ ] **Step 5: Run full API/engine tests**

Run: `pytest tests/test_engine.py tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 6: Verify token interval derivability**

Run: `pytest tests/test_api.py::test_stream_token_intervals_derivable -v`  
Expected: PASS with deterministic interval derivation from event timestamps.

- [ ] **Step 7: Commit**

```bash
git add src/core/model.py src/api/server.py src/core/config.py tests/test_engine.py tests/test_api.py
git commit -m "feat: add token-level streaming path and unified sse contract"
```

---

### Task 4: Add Profiling Pipeline and Report Normalization

**Files:**
- Create: `scripts/profile_inference.sh`
- Create: `docs/perf/profiles/README.md`
- Create: `docs/perf/baselines/<run-id>-summary.md` (generated output convention)
- Modify: `README.md`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing test**

```python
def test_profile_command_documented():
    with open("README.md", "r", encoding="utf-8") as f:
        text = f.read()
    assert "scripts/profile_inference.sh" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_profile_command_documented -v`  
Expected: FAIL before docs update.

- [ ] **Step 3: Write minimal implementation**

```bash
#!/usr/bin/env bash
nsys profile -o "${OUT_DIR}/${RUN_ID}" --trace=cuda,nvtx python scripts/benchmark_inference.py ...
```

Note: If `nsys` or GPU is unavailable, this step is marked "manual/perf-env only" and CI should skip it.

- [ ] **Step 4: Add operator hotspot categorization output**

Run: `bash scripts/profile_inference.sh --run-id <run_id>`  
Expected: summary includes explicit category shares for `attention/norm/mlp/sampling`.

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_api.py::test_profile_command_documented -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/profile_inference.sh docs/perf/profiles/README.md README.md tests/test_api.py
git commit -m "chore: add profiling pipeline and reporting docs"
```

---

### Task 5: Land M2 Triton Pilot (RMSNorm + Fallback)

**Files:**
- Create: `src/core/ops/triton_rmsnorm.py`
- Create: `src/core/ops/__init__.py`
- Modify: `src/core/model.py`
- Modify: `src/core/config.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_rmsnorm_triton_flag_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_RMSNORM", "true")
    out = safe_rmsnorm(x, w)
    assert out is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_rmsnorm_triton_flag_falls_back_when_unavailable -v`  
Expected: FAIL due to missing module or fallback path.

- [ ] **Step 3: Write minimal implementation**

```python
def safe_rmsnorm(x, w):
    try:
        return triton_rmsnorm(x, w)
    except Exception:
        return torch_rmsnorm_fallback(x, w)
```

- [ ] **Step 4: Add numeric threshold tests**

Run: `pytest tests/test_engine.py -k rmsnorm -v`  
Expected: includes tolerance assertions (`rtol<=1e-3`, `atol<=1e-4`).

- [ ] **Step 5: Commit**

```bash
git add src/core/ops/triton_rmsnorm.py src/core/ops/__init__.py src/core/model.py src/core/config.py tests/test_engine.py
git commit -m "feat: add triton rmsnorm pilot with runtime fallback"
```

---

### Task 6: Prepare M3 Decode-Attention Implementation Contract

**Files:**
- Create: `docs/perf/design/decode-attention-contract.md`
- Modify: `docs/superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md`
- Modify: `src/core/config.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write the failing test**

```python
def test_decode_attention_flag_exists(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", "true")
    assert load_runtime_flags().enable_decode_attn is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_engine.py::test_decode_attention_flag_exists -v`  
Expected: FAIL due to missing config flag.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class RuntimeOptFlags:
    enable_decode_attn: bool = _env_bool("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_engine.py::test_decode_attention_flag_exists -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/core/config.py tests/test_engine.py docs/perf/design/decode-attention-contract.md docs/superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md
git commit -m "chore: define decode attention optimization contract and flags"
```

---

### Task 7: M4 Decision Gate and Final Regression

**Files:**
- Create: `docs/perf/decisions/m4-go-no-go.md`
- Modify: `README.md`
- Test: `tests/test_api.py`
- Test: `tests/test_engine.py`

- [ ] **Step 1: Write failing test for decision threshold docs**

```python
def test_readme_mentions_m4_threshold():
    with open("README.md", "r", encoding="utf-8") as f:
        text = f.read()
    assert "热点占比超过 8%" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api.py::test_readme_mentions_m4_threshold -v`  
Expected: FAIL before docs update.

- [ ] **Step 3: Write minimal implementation**

```markdown
Only continue M4 when remaining hotspot exceeds 8% in profiler summary.
```

- [ ] **Step 4: Enforce switch-budget cap**

Run: `pytest tests/test_engine.py::test_optimization_switch_count_capped -v`  
Expected: PASS and asserts total optimization flags <= 3.

- [ ] **Step 5: Run final regression suite**

Run: `pytest tests/test_engine.py tests/test_api.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add docs/perf/decisions/m4-go-no-go.md README.md tests/test_api.py tests/test_engine.py
git commit -m "docs: add m4 go-no-go threshold and regression completion criteria"
```

---

## Global Verification Checklist (run after each task)

- [ ] Run targeted tests for touched files first.
- [ ] Run `pytest tests/test_engine.py tests/test_api.py -v` before milestone completion.
- [ ] Update baseline/profiling artifacts with `run_id`.
- [ ] Record deltas versus previous baseline in Markdown summary.
- [ ] Verify runtime fallback path by forcing Triton failure mode in at least one test.
- [ ] Record TTFT/TPOT/throughput deltas and mark pass/fail against hard gates.
- [ ] Ensure profiling report includes attention/norm/mlp/sampling category percentages.
- [ ] If 8-way concurrency is skipped, store explicit VRAM-based skip reason in report.

## Rollback Checklist (apply per task before merge)

- [ ] Confirm new optimization flags default to disabled.
- [ ] Confirm disabling env flags restores pre-change behavior.
- [ ] Document one-command rollback path (`unset` optimization envs + restart).
- [ ] If regression appears, revert only the current task commit (no cross-task rollback).

---

## Follow-On Plans Trigger

Create next two plans only when both gates pass:

- Gate A: M2 RMSNorm pilot meets spec thresholds against M0 baseline.
- Gate B: 24h stress run has no critical regression.

Required follow-on plans:

1. `docs/superpowers/plans/YYYY-MM-DD-m3-decode-attention-implementation.md`
2. `docs/superpowers/plans/YYYY-MM-DD-m4-mlp-sampling-fusion-implementation.md`

---

## Dependency and Execution Order

1. Task 1 -> Task 2 (baseline first, then richer metrics)
2. Task 2 -> Task 3 (token streaming depends on observability)
3. Task 3 -> Task 4 (profile pipeline depends on stable stream path)
4. Task 4 -> Task 5 (Triton pilot begins after profiling discipline)
5. Task 5 -> Task 6 (decode-attn contract after pilot proving pattern)
6. Task 6 -> Task 7 (final go/no-go and regression)

---

## Out of Scope (for this plan)

- Full vLLM scheduler re-architecture
- Quantization migration (AWQ/GPTQ/FP8)
- Multi-node distributed inference orchestration

