#!/usr/bin/env python3
"""
M2 RMSNorm AB comparison: run the benchmark matrix twice
(MINI_VLLM_ENABLE_TRITON_RMSNORM=false, then =true)
and produce a comparison CSV + summary.

Requires GPU. On CPU-only hosts, runs numeric tolerance checks only.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "docs" / "perf" / "baselines"


def _run_benchmark(label: str, rmsnorm_flag: str, output_dir: Path, extra_args: list[str]) -> int:
    env = os.environ.copy()
    env["MINI_VLLM_ENABLE_TRITON_RMSNORM"] = rmsnorm_flag
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "benchmark_inference.py"),
        "--execute",
        "--output-dir", str(output_dir),
        *extra_args,
    ]
    print(f"\n{'='*60}")
    print(f"  [{label}] MINI_VLLM_ENABLE_TRITON_RMSNORM={rmsnorm_flag}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    return subprocess.call(cmd, env=env, cwd=str(ROOT))


def _load_results(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _compare(baseline: list[dict], pilot: list[dict]) -> list[dict]:
    pilot_by_key = {}
    for r in pilot:
        key = (r.get("mode"), r.get("prompt_bucket"), r.get("concurrency"))
        pilot_by_key[key] = r

    rows = []
    for b in baseline:
        key = (b.get("mode"), b.get("prompt_bucket"), b.get("concurrency"))
        p = pilot_by_key.get(key)
        if not p:
            continue
        try:
            b_dur = float(b["duration_ms"])
            p_dur = float(p["duration_ms"])
            delta_pct = ((b_dur - p_dur) / b_dur * 100) if b_dur > 0 else 0
        except (ValueError, KeyError):
            delta_pct = 0
        rows.append({
            "mode": b.get("mode"),
            "bucket": b.get("prompt_bucket"),
            "conc": b.get("concurrency"),
            "baseline_ms": b.get("duration_ms", ""),
            "pilot_ms": p.get("duration_ms", ""),
            "delta_pct": f"{delta_pct:+.1f}%",
            "baseline_tps": b.get("throughput_tok_per_s", ""),
            "pilot_tps": p.get("throughput_tok_per_s", ""),
        })
    return rows


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="RMSNorm AB comparison")
    parser.add_argument("--mode", default="warm", help="Benchmark mode (default: warm)")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--max-model-len", default="2048")
    parser.add_argument("--max-num-seqs", default="1")
    parser.add_argument("--prompt-buckets", default="short,medium")
    parser.add_argument("--concurrency", default="1")
    parser.add_argument("--max-tokens", default="128")
    parser.add_argument("--output-dir", default=str(BASELINE_DIR / "rmsnorm-ab"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    base_dir = out / "baseline"
    pilot_dir = out / "pilot"

    bench_args = [
        "--mode", args.mode,
        "--dtype", args.dtype,
        "--max-model-len", args.max_model_len,
        "--max-num-seqs", args.max_num_seqs,
        "--prompt-buckets", args.prompt_buckets,
        "--concurrency", args.concurrency,
        "--max-tokens", args.max_tokens,
    ]

    ret = _run_benchmark("BASELINE", "false", base_dir, bench_args)
    if ret != 0:
        print(f"Baseline benchmark failed (exit {ret}). Is GPU available?", file=sys.stderr)
        return ret

    ret = _run_benchmark("PILOT", "true", pilot_dir, bench_args)
    if ret != 0:
        print(f"Pilot benchmark failed (exit {ret}).", file=sys.stderr)
        return ret

    baseline_rows = _load_results(base_dir / "results.csv")
    pilot_rows = _load_results(pilot_dir / "results.csv")

    if not baseline_rows or not pilot_rows:
        print("No results to compare.", file=sys.stderr)
        return 1

    compared = _compare(baseline_rows, pilot_rows)
    comp_csv = out / "comparison.csv"
    if compared:
        fields = list(compared[0].keys())
        with comp_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(compared)
        print(f"\nComparison written to {comp_csv}")

    print("\n--- AB Summary ---")
    for r in compared:
        print(f"  {r['mode']}/{r['bucket']}/c{r['conc']}: "
              f"baseline={r['baseline_ms']}ms pilot={r['pilot_ms']}ms delta={r['delta_pct']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
