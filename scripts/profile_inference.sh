#!/usr/bin/env bash
# Profile inference benchmark path with Nsight Systems (nsys), or emit stub hotspot summary.
# Supports --execute to run real inference under nsys instead of --dry-run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=""
OUT_DIR=""
EXECUTE=false
MAX_TOKENS=128
BENCH_ARGS=""

usage() {
  echo "Usage: $0 --run-id <id> [--output-dir <dir>] [--execute] [--max-tokens N] [-- extra benchmark_inference.py args]" >&2
  echo "  --run-id       Required. Names nsys output and hotspot summary files." >&2
  echo "  --output-dir   Directory for raw nsys reports (default: docs/perf/profiles/nsys)." >&2
  echo "  --execute      Pass --execute to benchmark_inference.py (real inference, needs GPU)." >&2
  echo "  --max-tokens   Max tokens per scenario (default: 128). Only used with --execute." >&2
  echo "  --             Remaining args are forwarded to benchmark_inference.py." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --execute)
      EXECUTE=true
      shift
      ;;
    --max-tokens)
      MAX_TOKENS="${2:-128}"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      BENCH_ARGS="$*"
      break
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$RUN_ID" ]]; then
  echo "error: --run-id is required" >&2
  usage
  exit 2
fi

if [[ -z "${OUT_DIR}" ]]; then
  OUT_DIR="docs/perf/profiles/nsys"
fi

SUMMARY_PATH="docs/perf/profiles/${RUN_ID}-hotspot-summary.md"
mkdir -p "$(dirname "$SUMMARY_PATH")"
mkdir -p "${OUT_DIR}/${RUN_ID}"

write_stub_summary() {
  cat > "$SUMMARY_PATH" <<EOF
# Hotspot summary (\`${RUN_ID}\`)

> **Environment:** \`nsys\` was not found on \`PATH\`. The shares below are **stubs**. Install [Nsight Systems](https://developer.nvidia.com/nsight-systems) and ensure \`nsys\` is on \`PATH\`, then re-run this script on a GPU host to capture a real profile.

## attention

**Percent share:** 0% (stub — N/A — run with nsys)

## norm

**Percent share:** 0% (stub — N/A — run with nsys)

## mlp

**Percent share:** 0% (stub — N/A — run with nsys)

## sampling

**Percent share:** 0% (stub — N/A — run with nsys)

## Notes

- Replace stub percentages using the Nsight Systems report (GPU kernel / NVTX time breakdown) after a successful \`nsys profile\` run.
- CI or a manual GPU environment should populate real percentages from the exported \`.nsys-rep\` / summary views.
- Optional: pair with \`docs/perf/baselines/<run-id>-summary.md\` and benchmark CSV for the same \`run_id\`.
EOF
}

write_nsys_placeholder_summary() {
  local nsys_out_rel="${OUT_DIR}/${RUN_ID}/nsys"
  local mode_note
  if [[ "$EXECUTE" == "true" ]]; then
    mode_note="> **Mode:** \`--execute\` — real GPU inference was profiled. The hotspot percentages below reflect actual kernel execution."
  else
    mode_note="> **Limitation — \`--dry-run\`:** The profile reflects **Python / orchestration overhead** only. Re-run with \`--execute\` on a GPU host for kernel-level hotspots."
  fi
  cat > "$SUMMARY_PATH" <<EOF
# Hotspot summary (\`${RUN_ID}\`)

> **Source:** \`nsys profile\` completed. Raw report prefix: \`${nsys_out_rel}\` (Nsight Systems adds \`.nsys-rep\` / \`.qdstrm\` beside this prefix).

${mode_note}

## attention

**Percent share (placeholder):** _Open the report in Nsight Systems and fill GPU time % for attention-related work (kernels / NVTX ranges)._

## norm

**Percent share (placeholder):** _Fill from Nsight Systems UI._

## mlp

**Percent share (placeholder):** _Fill from Nsight Systems UI._

## sampling

**Percent share (placeholder):** _Fill from Nsight Systems UI._

## Notes

- Automated parsing of \`nsys\` output is not performed here; copy percentages from the Nsight Systems **Summary** / **Timeline** views.
- Optional: pair with \`docs/perf/baselines/<run-id>-summary.md\` and the benchmark CSV for the same \`run_id\`.
EOF
}

if ! command -v nsys >/dev/null 2>&1; then
  write_stub_summary
  exit 0
fi

BENCH_RESULTS_DIR="${OUT_DIR}/${RUN_ID}"

if [[ "$EXECUTE" == "true" ]]; then
  DEFAULT_BENCH_ARGS="--execute --max-tokens ${MAX_TOKENS} --mode warm --dtype fp16 --max-model-len 2048 --max-num-seqs 1 --prompt-buckets short,medium --concurrency 1 --output-dir ${BENCH_RESULTS_DIR}"
  if [[ -n "$BENCH_ARGS" ]]; then
    FINAL_ARGS="--execute --max-tokens ${MAX_TOKENS} ${BENCH_ARGS}"
  else
    FINAL_ARGS="$DEFAULT_BENCH_ARGS"
  fi
else
  FINAL_ARGS="--dry-run --mode warm --dtype fp16 --max-model-len 2048 --max-num-seqs 1 --prompt-buckets short --concurrency 1"
fi

nsys profile \
  -o "${OUT_DIR}/${RUN_ID}/nsys" \
  --trace=cuda,nvtx \
  --force-overwrite true \
  -- \
  python scripts/benchmark_inference.py ${FINAL_ARGS}

write_nsys_placeholder_summary
exit 0
