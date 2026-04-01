"""
Shared helpers for inference benchmark runs (run_id, scenario dimensions, parsing).
"""
import re
import time
from typing import Any

# Representative prompt *character* lengths per bucket (simple proxy for token-ish load).
PROMPT_BUCKET_CHAR_LENGTHS: dict[str, int] = {
    "short": 512,
    "medium": 4096,
    "long": 16384,
}


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
