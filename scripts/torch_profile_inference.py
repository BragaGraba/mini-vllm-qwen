#!/usr/bin/env python3
"""
Profile inference with torch.profiler to identify GPU kernel hotspots.
Categorizes kernels into: attention, norm, mlp, sampling, other.
Use when nsys is not available.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from src.core.benchmarking import build_prompt_for_bucket
from src.core.model import MiniVLLMEngine


CATEGORY_KEYWORDS = {
    "attention": ["attention", "attn", "flash", "sdpa", "softmax", "bmm", "baddbmm"],
    "norm": ["norm", "rmsnorm", "layernorm", "layer_norm", "rms_norm"],
    "mlp": ["mlp", "gate", "silu", "gelu", "linear", "gemm", "mm", "addmm", "matmul"],
    "sampling": ["sample", "argmax", "topk", "top_k", "multinomial", "softmax_out"],
}


def categorize_kernel(name: str) -> str:
    lower = name.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                return cat
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile inference with torch.profiler")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", default="short", help="Prompt bucket: short/medium")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output-dir", default="docs/perf/profiles")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--profile-runs", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = build_prompt_for_bucket(args.bucket)
    engine = MiniVLLMEngine(max_model_len=2048, max_num_seqs=1, dtype="float16")

    print(f"Warming up ({args.warmup_runs} runs)...")
    for _ in range(args.warmup_runs):
        engine.generate(prompt, stream=False, max_tokens=args.max_tokens, temperature=0.0)
    print("Warmup done.")

    print(f"Profiling ({args.profile_runs} runs)...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for _ in range(args.profile_runs):
            with record_function("inference"):
                engine.generate(prompt, stream=False, max_tokens=args.max_tokens, temperature=0.0)

    cuda_events = [
        e for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA or (hasattr(e, 'cuda_time_total') and e.cuda_time_total > 0)
    ]

    category_time: dict[str, float] = {"attention": 0, "norm": 0, "mlp": 0, "sampling": 0, "other": 0}
    for event in cuda_events:
        ct = event.cuda_time_total if hasattr(event, 'cuda_time_total') else 0
        if ct > 0:
            cat = categorize_kernel(event.key)
            category_time[cat] += ct

    total_cuda = sum(category_time.values())
    category_pct = {}
    if total_cuda > 0:
        category_pct = {k: round(v / total_cuda * 100, 1) for k, v in category_time.items()}
    else:
        category_pct = {k: 0.0 for k in category_time}

    summary_path = out_dir / f"{args.run_id}-hotspot-summary.md"
    with summary_path.open("w") as f:
        f.write(f"# Hotspot summary (`{args.run_id}`)\n\n")
        f.write(f"> **Source:** `torch.profiler` on RTX 3060 12GB, CUDA 12.8\n")
        f.write(f"> **Bucket:** `{args.bucket}`, **max_tokens:** {args.max_tokens}, ")
        f.write(f"**warmup:** {args.warmup_runs}, **profile runs:** {args.profile_runs}\n\n")

        for cat in ["attention", "norm", "mlp", "sampling", "other"]:
            pct = category_pct.get(cat, 0)
            time_us = category_time.get(cat, 0)
            f.write(f"## {cat}\n\n")
            f.write(f"**Percent share:** {pct}% ({time_us/1000:.1f}ms CUDA time)\n\n")

        f.write("## Top 20 CUDA kernels\n\n")
        f.write("| Kernel | Category | CUDA Time (ms) | Calls |\n")
        f.write("|--------|----------|---------------|-------|\n")
        sorted_events = sorted(cuda_events, key=lambda e: getattr(e, 'cuda_time_total', 0), reverse=True)[:20]
        for e in sorted_events:
            ct = getattr(e, 'cuda_time_total', 0)
            count = e.count if hasattr(e, 'count') else 0
            cat = categorize_kernel(e.key)
            name = e.key[:60]
            f.write(f"| `{name}` | {cat} | {ct/1000:.2f} | {count} |\n")

        f.write(f"\n## Notes\n\n")
        f.write(f"- Total CUDA time: {total_cuda/1000:.1f}ms\n")
        f.write(f"- Categorization is keyword-based (see `CATEGORY_KEYWORDS` in script)\n")
        f.write(f"- Some kernels (e.g. element-wise ops, embedding) fall into 'other'\n")

    print(f"\nHotspot summary written to: {summary_path}")
    print(f"\nCategory breakdown:")
    for cat in ["attention", "norm", "mlp", "sampling", "other"]:
        print(f"  {cat:12s}: {category_pct.get(cat, 0):5.1f}% ({category_time.get(cat, 0)/1000:.1f}ms)")

    json_path = out_dir / f"{args.run_id}-profile-data.json"
    json_data = {
        "run_id": args.run_id,
        "bucket": args.bucket,
        "max_tokens": args.max_tokens,
        "total_cuda_time_us": total_cuda,
        "category_time_us": category_time,
        "category_pct": category_pct,
    }
    json_path.write_text(json.dumps(json_data, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
