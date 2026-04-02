# M3 A/B — vLLM `TRITON_ATTN` vs default attention

When **`MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=true`**, `MiniVLLMEngine` passes
`AttentionConfig(backend=TRITON_ATTN)` to `vllm.LLM`. When **false** (unset), vLLM auto-selects
the default backend (e.g. FlashAttention on NVIDIA).

## Commands (minimal matrix)

```bash
cd /path/to/mini-vllm-qwen

# Baseline (auto backend)
MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=false \
  .venv/bin/python scripts/benchmark_inference.py --execute \
  --mode warm --dtype float16 --max-model-len 2048 --max-num-seqs 1 \
  --prompt-buckets short --concurrency 1 --max-tokens 64 \
  --output-dir docs/perf/baselines/m3-ab/flash-default

# Pilot (Triton attention backend)
MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=true \
  .venv/bin/python scripts/benchmark_inference.py --execute \
  --mode warm --dtype float16 --max-model-len 2048 --max-num-seqs 1 \
  --prompt-buckets short --concurrency 1 --max-tokens 64 \
  --output-dir docs/perf/baselines/m3-ab/triton-attn
```

Compare `results.csv` in each directory (same `run_id` scope per invocation — note `run_id` differs between runs).

## Latest run snapshot (2026-04-02, RTX 3060 12GB, Qwen2.5-3B-Instruct, WSL2)

| Mode | Attention | duration_ms (warm/short/c1) | throughput_tok_per_s | run_id |
|------|-----------|------------------------------|----------------------|--------|
| baseline | auto (FlashAttn selected by vLLM) | 17112.5 | 5.73 | `run-qwen-float16-1775094837` |
| pilot | TRITON_ATTN | 15479.1 | 6.33 | `run-qwen-float16-1775094861` |

**Note:** These one-row runs include engine load + one generate; variance is high. For Gate decisions, use a larger matrix and repeat runs (see main baseline report).

## Gate (M3)

- Re-check design §4.2 (TTFT/TPOT/throughput vs M0 anchor run_id) using the same workload.
- If TRITON_ATTN regresses, set `MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=false` and file an issue with logs + `run_metadata.json`.
