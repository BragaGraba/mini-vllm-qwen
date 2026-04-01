# Inference profiling (Nsight Systems)

This folder holds **hotspot summaries** and (by default) **raw Nsight Systems** outputs from `scripts/profile_inference.sh`.

## Requirements

- **GPU profiling:** Meaningful CUDA kernel data requires a **Linux host with an NVIDIA GPU**, Nsight Systems installed, and `nsys` on `PATH`. WSL2 may work when GPU passthrough and Nsight are set up; otherwise treat profiling as **manual / perf-lab only** — CI can skip it.

## How to run

From the repository root:

```bash
chmod +x scripts/profile_inference.sh   # once
./scripts/profile_inference.sh --run-id my-run-001
```

Options:

| Option | Description |
|--------|-------------|
| `--run-id <id>` | **Required.** Used for output naming (see below). |
| `--output-dir <dir>` | Where to store raw `nsys` reports. Default: `docs/perf/profiles/nsys`. |

### Outputs

1. **Raw profile (when `nsys` is available):**  
   `${output-dir}/${run-id}/nsys` (Nsight appends `.nsys-rep` / related files).

2. **Hotspot summary (always written):**  
   `docs/perf/profiles/<run-id>-hotspot-summary.md`  
   Sections: **attention**, **norm**, **mlp**, **sampling** (percent shares). If `nsys` is missing, stub values are written and the script exits **0**. If `nsys` ran successfully, placeholder text is written — fill real percentages from the Nsight Systems UI unless you add automated parsing later.

### Dry-run limitation

The script invokes `python scripts/benchmark_inference.py` with **`--dry-run`** and the **smallest** scenario matrix so profiling finishes quickly. That exercises **Python / matrix setup** only, not GPU inference kernels. For representative GPU hotspots, profile a real workload (future `--execute`, or the running API) and adjust arguments as needed.

## Pairing with baselines

Benchmark CSV exports and metadata live under [../baselines/](../baselines/README.md). A useful convention is to add an optional human-readable **`docs/perf/baselines/<run-id>-summary.md`** next to `scenarios.csv` / `run_metadata.json` for the same **`run_id`**, so profiling notes and benchmark tables stay aligned.

## Related

- Baseline exports: [../baselines/README.md](../baselines/README.md)
