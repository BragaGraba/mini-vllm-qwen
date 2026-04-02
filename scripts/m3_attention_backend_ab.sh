#!/usr/bin/env bash
# Run minimal M3 A/B: default attention vs TRITON_ATTN (same benchmark matrix).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${ROOT}/.venv/bin/python"

FLASH_OUT="${ROOT}/docs/perf/baselines/m3-ab/flash-default"
TRITON_OUT="${ROOT}/docs/perf/baselines/m3-ab/triton-attn"

BENCH=(
  scripts/benchmark_inference.py --execute
  --mode warm --dtype float16
  --max-model-len 2048 --max-num-seqs 1
  --prompt-buckets short --concurrency 1 --max-tokens 64
)

echo "=== M3 A/B: Flash/auto baseline ==="
MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=false \
  "$PY" "${BENCH[@]}" --output-dir "$FLASH_OUT"

echo "=== M3 A/B: TRITON_ATTN pilot ==="
MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=true \
  "$PY" "${BENCH[@]}" --output-dir "$TRITON_OUT"

echo "Done. Compare:"
echo "  $FLASH_OUT/results.csv"
echo "  $TRITON_OUT/results.csv"
