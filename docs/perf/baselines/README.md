# M0 inference benchmark baselines

This folder is the documented home for benchmark CSV exports and `run_metadata.json` files produced by `scripts/benchmark_inference.py`. You can point outputs here with `--output-dir` or the `MINI_VLLM_BENCHMARK_OUT` environment variable.

Design context: [§6 M0 / §2.4 实现状态摘要](../../superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md#implementation-status).

**实测报告骨架（复制后按 `run_id` 重命名填写）：** [`TEMPLATE-run-report.md`](TEMPLATE-run-report.md)

**M3 attention A/B：** [`m3-triton-attn-ab.md`](m3-triton-attn-ab.md)（产物目录 `m3-ab/`）

## Tests

Install dev dependencies (includes `pytest`):

```bash
pip install -r requirements-dev.txt
```

## Prompt buckets (fixed sizes)

Each bucket maps to a **representative character length** used for workload sizing in the scenario matrix (simple proxy for “short / medium / long” prompts):

| Bucket | Character length |
|--------|------------------|
| `short` | 512 |
| `medium` | 4096 |
| `long` | 16384 |

Defined in `src/core/benchmarking.PROMPT_BUCKET_CHAR_LENGTHS`.

## `run_id` scope

Each script invocation generates **one** `run_id` via `build_run_id(model, first_dtype)` and reuses it for **every** scenario row in that run (Cartesian product of all dimensions). The `dtype` segment in `run_id` uses the **first** value in the `--dtype` comma list when multiple dtypes are swept.

## CLI help

```bash
python scripts/benchmark_inference.py --help
```

## Examples (plan Steps 6–7)

**Step 6 — cold/warm and core dimensions (NDJSON on stdout):**

```bash
python scripts/benchmark_inference.py --dry-run \
  --mode cold,warm \
  --dtype fp16 \
  --max-model-len 2048,4096 \
  --max-num-seqs 1,2,4 \
  --prompt-buckets short,medium,long \
  --concurrency 1,2,4
```

**Step 7 — full concurrency sweep including 8, with an explicit skip reason for VRAM:**

```bash
python scripts/benchmark_inference.py --dry-run \
  --prompt-buckets short,medium,long \
  --concurrency 1,2,4,8 \
  --mode cold,warm \
  --dtype fp16 \
  --max-model-len 2048,4096 \
  --max-num-seqs 1,2,4 \
  --skip-concurrency-reason "8: insufficient VRAM"
```

Equivalent via environment (no GPU required for scenario generation):

```bash
export MINI_VLLM_SKIP_CONCURRENCY=8
export MINI_VLLM_SKIP_CONCURRENCY_REASON="insufficient VRAM"
python scripts/benchmark_inference.py --dry-run \
  --prompt-buckets short,medium,long \
  --concurrency 1,2,4,8 \
  --mode cold,warm \
  --dtype fp16 \
  --max-model-len 2048,4096 \
  --max-num-seqs 1,2,4
```

CLI `--skip-concurrency-reason` takes precedence over the env mapping when both target the same concurrency (env fills in only if the concurrency key is not already set).

## Outputs (`--output-dir` / `MINI_VLLM_BENCHMARK_OUT`)

With `--output-dir PATH` (or `MINI_VLLM_BENCHMARK_OUT=PATH`), the script writes:

- `scenarios.csv` — one row per scenario (same fields as NDJSON lines, including optional `skipped_reason`).
- `run_metadata.json` — `run_id`, UTC timestamp, argv, row count, and a note that `run_id` is shared across rows.

Example:

```bash
python scripts/benchmark_inference.py --dry-run --output-dir docs/perf/baselines \
  --mode cold,warm \
  --dtype fp16 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --prompt-buckets short \
  --concurrency 1
```

You must pass **`--dry-run` and/or an output directory**; otherwise the script exits with an error reminding you to choose one of those.

## Related configuration

| Variable | Purpose |
|----------|---------|
| `MINI_VLLM_BENCHMARK_OUT` | Default `--output-dir` when set |
| `MINI_VLLM_SKIP_CONCURRENCY` | Concurrency level to mark as skipped (e.g. `8`) |
| `MINI_VLLM_SKIP_CONCURRENCY_REASON` | Reason string paired with the env skip |
| `MINI_VLLM_MODEL_NAME` | Default `--model` label for `run_id` sanitization |

See `src/core/config.py` for helpers.
