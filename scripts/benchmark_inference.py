#!/usr/bin/env python3
"""
M0 inference benchmark scenario matrix: run_id, dimensions, optional skip reasons.
Real GPU execution is optional/future; default is scenario listing (NDJSON) or CSV export.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.benchmarking import build_run_id, parse_comma_list, scenario_row  # noqa: E402
from src.core.config import get_benchmark_output_dir, get_skip_concurrency_env  # noqa: E402


def _parse_skip_concurrency_reason_arg(raw: str | None) -> dict[int, str]:
    """Parse '8: insufficient VRAM' -> {8: 'insufficient VRAM'}."""
    if not raw or not raw.strip():
        return {}
    text = raw.strip()
    if ":" not in text:
        return {}
    left, right = text.split(":", 1)
    try:
        n = int(left.strip())
    except ValueError:
        return {}
    reason = right.strip() or "skipped"
    return {n: reason}


def _merge_skip_reasons(cli_map: dict[int, str], env_n: int | None, env_reason: str) -> dict[int, str]:
    out = dict(cli_map)
    if env_n is not None and env_n > 0:
        out.setdefault(env_n, env_reason)
    return out


def _nonempty_list(label: str, items: list) -> list:
    if not items:
        print(f"error: {label} must resolve to at least one value", file=sys.stderr)
        raise SystemExit(2)
    return items


def _parse_comma_ints(label: str, raw: str) -> list[int]:
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            print(f"error: {label} contains non-integer token {p!r}", file=sys.stderr)
            raise SystemExit(2) from None
    return out


def _iter_scenarios(
    *,
    run_id: str,
    modes: list[str],
    dtypes: list[str],
    max_model_lens: list[int],
    max_num_seqs_list: list[int],
    prompt_buckets: list[str],
    concurrencies: list[int],
    skip_by_concurrency: dict[int, str],
) -> list[dict]:
    rows: list[dict] = []
    for mode, dtype, mml, mns, bucket, conc in product(
        modes,
        dtypes,
        max_model_lens,
        max_num_seqs_list,
        prompt_buckets,
        concurrencies,
    ):
        skipped = skip_by_concurrency.get(conc)
        rows.append(
            scenario_row(
                run_id,
                mode,
                dtype,
                mml,
                mns,
                bucket,
                conc,
                skipped_reason=skipped,
            )
        )
    return rows


def main() -> int:
    env_out = get_benchmark_output_dir()
    parser = argparse.ArgumentParser(
        description=(
            "Build a full Cartesian benchmark matrix (modes × dtype × context × buckets × concurrency). "
            "Prompt buckets use fixed representative character lengths (see --help). "
            "One shared run_id is generated per invocation for all scenario rows."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MINI_VLLM_MODEL_NAME", "qwen"),
        help="Model label for run_id sanitization (default: qwen or MINI_VLLM_MODEL_NAME).",
    )
    parser.add_argument("--mode", required=True, help="Comma-separated modes, e.g. cold,warm")
    parser.add_argument("--dtype", required=True, help="Comma-separated dtypes, e.g. fp16,bf16")
    parser.add_argument(
        "--max-model-len",
        required=True,
        help="Comma-separated ints, e.g. 2048,4096",
    )
    parser.add_argument(
        "--max-num-seqs",
        required=True,
        help="Comma-separated ints, e.g. 1,2,4",
    )
    parser.add_argument(
        "--prompt-buckets",
        required=True,
        help="Comma-separated: short,medium,long (fixed char lengths: short=512, medium=4096, long=16384)",
    )
    parser.add_argument(
        "--concurrency",
        required=True,
        help="Comma-separated ints, e.g. 1,2,4,8",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print one JSON object per scenario line (NDJSON) to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        default=env_out or None,
        help="Write scenarios.csv + run_metadata.json (optional; also MINI_VLLM_BENCHMARK_OUT).",
    )
    parser.add_argument(
        "--skip-concurrency-reason",
        default=None,
        metavar="N: REASON",
        help='Mark concurrency N as skipped with reason, e.g. "8: insufficient VRAM".',
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run real inference benchmark (not implemented in M0; exits with error).",
    )
    args = parser.parse_args()

    modes = _nonempty_list("--mode", parse_comma_list(args.mode))
    dtypes = _nonempty_list("--dtype", parse_comma_list(args.dtype))
    max_model_lens = _nonempty_list("--max-model-len", _parse_comma_ints("--max-model-len", args.max_model_len))
    max_num_seqs_list = _nonempty_list(
        "--max-num-seqs",
        _parse_comma_ints("--max-num-seqs", args.max_num_seqs),
    )
    prompt_buckets = _nonempty_list(
        "--prompt-buckets",
        [x.lower() for x in parse_comma_list(args.prompt_buckets)],
    )
    concurrencies = _nonempty_list("--concurrency", _parse_comma_ints("--concurrency", args.concurrency))

    run_id = build_run_id(args.model, dtypes[0])

    cli_skip = _parse_skip_concurrency_reason_arg(args.skip_concurrency_reason)
    env_n, env_reason = get_skip_concurrency_env()
    skip_by_concurrency = _merge_skip_reasons(cli_skip, env_n, env_reason)

    rows = _iter_scenarios(
        run_id=run_id,
        modes=modes,
        dtypes=dtypes,
        max_model_lens=max_model_lens,
        max_num_seqs_list=max_num_seqs_list,
        prompt_buckets=prompt_buckets,
        concurrencies=concurrencies,
        skip_by_concurrency=skip_by_concurrency,
    )

    out_dir = args.output_dir
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        csv_path = p / "scenarios.csv"
        fieldnames = [
            "run_id",
            "mode",
            "dtype",
            "max_model_len",
            "max_num_seqs",
            "prompt_bucket",
            "prompt_char_len",
            "concurrency",
            "skipped_reason",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                line = {k: r.get(k, "") for k in fieldnames}
                if line["skipped_reason"] is None:
                    line["skipped_reason"] = ""
                w.writerow(line)
        meta = {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "row_count": len(rows),
            "run_id_scope": "one shared run_id per script invocation for all scenario rows",
        }
        (p / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.execute:
        print("Real benchmark execution is not implemented in the M0 framework.", file=sys.stderr)
        return 1

    if args.dry_run:
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
    elif not out_dir:
        print(
            "Specify --dry-run (NDJSON on stdout) and/or --output-dir "
            "(or set MINI_VLLM_BENCHMARK_OUT).",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
