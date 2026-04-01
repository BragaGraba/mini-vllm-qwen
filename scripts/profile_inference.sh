#!/usr/bin/env bash
# Profile inference benchmark path with Nsight Systems (nsys), or emit stub hotspot summary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID=""
OUT_DIR=""

usage() {
  echo "Usage: $0 --run-id <id> [--output-dir <dir>]" >&2
  echo "  --run-id       Required. Names nsys output and hotspot summary files." >&2
  echo "  --output-dir   Directory for raw nsys reports (default: docs/perf/profiles/nsys)." >&2
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
    -h | --help)
      usage
      exit 0
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
  cat > "$SUMMARY_PATH" <<EOF
# Hotspot summary (\`${RUN_ID}\`)

> **Source:** \`nsys profile\` completed. Raw report prefix: \`${nsys_out_rel}\` (Nsight Systems adds \`.nsys-rep\` / \`.qdstrm\` beside this prefix).

> **Limitation — \`--dry-run\`:** The wrapped command runs \`scripts/benchmark_inference.py\` with \`--dry-run\` and the **smallest** scenario matrix so the run finishes quickly. That path builds the benchmark matrix in Python only and does **not** execute GPU inference kernels. The profile therefore reflects **Python / orchestration overhead**, not a representative GPU hotspot split. For kernel-level hotspots, profile a real inference workload (e.g. once \`--execute\` exists, or by profiling the live API server) and pass any extra \`benchmark_inference.py\` arguments you need.

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

# Minimal matrix + dry-run: fast, no GPU inference (see limitation in summary).
nsys profile \
  -o "${OUT_DIR}/${RUN_ID}/nsys" \
  --trace=cuda,nvtx \
  --force-overwrite true \
  -- \
  python scripts/benchmark_inference.py \
  --dry-run \
  --mode warm \
  --dtype fp16 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --prompt-buckets short \
  --concurrency 1

write_nsys_placeholder_summary
exit 0
