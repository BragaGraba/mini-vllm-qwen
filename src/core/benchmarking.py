"""
Shared helpers for inference benchmark runs (run_id, scenario dimensions, parsing, execution).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

PROMPT_BUCKET_CHAR_LENGTHS: dict[str, int] = {
    "short": 512,
    "medium": 4096,
    "long": 16384,
}

RESULTS_CSV_FIELDS: list[str] = [
    "run_id",
    "mode",
    "dtype",
    "max_model_len",
    "max_num_seqs",
    "prompt_bucket",
    "prompt_char_len",
    "concurrency",
    "skipped_reason",
    "duration_ms",
    "completion_len",
    "completion_tokens_est",
    "ttft_ms",
    "tpot_ms",
    "throughput_tok_per_s",
    "error",
]


@dataclass
class TimingResult:
    duration_ms: float = 0.0
    completion_len: int = 0
    completion_tokens_est: int = 0
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    throughput_tok_per_s: float = 0.0
    error: str = ""


def build_prompt_for_bucket(bucket: str) -> str:
    """Build a synthetic prompt of the target character length for a given bucket."""
    target = PROMPT_BUCKET_CHAR_LENGTHS.get(bucket, 512)
    base = f"Please write a detailed explanation about topic-{bucket}. "
    if len(base) >= target:
        return base[:target]
    repeats = (target // len(base)) + 1
    return (base * repeats)[:target]


def run_single_scenario(
    engine: Any,
    prompt: str,
    *,
    max_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> TimingResult:
    """Execute one inference call and collect timing."""
    result = TimingResult()
    t0 = time.perf_counter()
    try:
        text = engine.generate(
            prompt,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if not isinstance(text, str):
            text = "".join(text)
    except Exception as exc:
        result.error = str(exc)
        result.duration_ms = (time.perf_counter() - t0) * 1000.0
        return result

    result.duration_ms = (time.perf_counter() - t0) * 1000.0
    result.completion_len = len(text)
    result.completion_tokens_est = max(1, len(text) // 4)
    ct = result.completion_tokens_est
    result.ttft_ms = result.duration_ms
    result.tpot_ms = result.duration_ms / ct if ct else result.duration_ms
    dur_s = result.duration_ms / 1000.0
    result.throughput_tok_per_s = ct / dur_s if dur_s > 0 else 0.0
    return result


def _sanitize_model_segment(model: str) -> str:
    s = model.strip().lower()
    if not s:
        return "model"
    out: list[str] = []
    for c in s:
        if c in "/\\ ":
            out.append("_")
        elif c.isalnum() or c in "._-":
            out.append(c)
        else:
            out.append("_")
    collapsed = re.sub(r"_+", "_", "".join(out)).strip("_")
    return collapsed or "model"


def build_run_id(model: str, dtype: str) -> str:
    ts = int(time.time())
    sanitized = _sanitize_model_segment(model)
    d = dtype.strip().lower() or "dtype"
    return f"run-{sanitized}-{d}-{ts}"


def parse_comma_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_comma_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def scenario_row(
    run_id: str,
    mode: str,
    dtype: str,
    max_model_len: int,
    max_num_seqs: int,
    prompt_bucket: str,
    concurrency: int,
    skipped_reason: str | None = None,
) -> dict[str, Any]:
    char_len = PROMPT_BUCKET_CHAR_LENGTHS.get(prompt_bucket, 0)
    row: dict[str, Any] = {
        "run_id": run_id,
        "mode": mode,
        "dtype": dtype,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "prompt_bucket": prompt_bucket,
        "prompt_char_len": char_len,
        "concurrency": concurrency,
    }
    parts: list[str] = []
    if skipped_reason:
        parts.append(skipped_reason)
    if char_len == 0 and prompt_bucket:
        parts.append(f"unknown prompt_bucket {prompt_bucket!r}")
    if parts:
        row["skipped_reason"] = "; ".join(parts)
    return row
