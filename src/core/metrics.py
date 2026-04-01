"""
轻量级监控：记录最近一段时间内的调用统计信息。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from fractions import Fraction
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional


def _percentile_ms(values: List[float], pct: float) -> float:
    """Linear interpolation percentile over ``values`` (0--100), e.g. pct=50 -> median."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n == 1:
        return float(s[0])
    # Use Fraction so (e.g.) 95th of 3 points lands at index 1.9 with exact 0.9 weight.
    rank = Fraction(int(round(pct * 1000)), 100000) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    w = float(rank - lo)
    return float(s[lo] + (s[hi] - s[lo]) * w)


@dataclass
class _Record:
    ts: float
    duration_ms: float
    prompt_len: int
    completion_len: int
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    token_window_ms: Optional[float] = None
    completion_tokens: Optional[int] = None


class BasicMetrics:
    """
    仅在内存中维护最近 N 条生成调用记录，用于 /metrics/basic。
    不依赖外部存储，重启即清空。
    """

    def __init__(self, max_records: int = 200) -> None:
        self._records: Deque[_Record] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def clear(self) -> None:
        """清空记录（供测试隔离使用）。"""
        with self._lock:
            self._records.clear()

    def record(
        self,
        *,
        duration_ms: float,
        prompt_len: int,
        completion_len: int,
        ttft_ms: float | None = None,
        tpot_ms: float | None = None,
        prefill_ms: float | None = None,
        decode_ms: float | None = None,
        first_token_ms: float | None = None,
        token_window_ms: float | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """
        记录一次生成调用。

        ``token_window_ms``：可选的按窗口/滚动统计用时（毫秒），例如在流式场景下表示
        每 N 个输出 token 的平均间隔（ms/token），或单次请求内按 token 维度的平均间隔；
        具体含义由调用方约定，仅做聚合平均。
        """
        with self._lock:
            self._records.append(
                _Record(
                    ts=time.time(),
                    duration_ms=duration_ms,
                    prompt_len=prompt_len,
                    completion_len=completion_len,
                    ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    first_token_ms=first_token_ms,
                    token_window_ms=token_window_ms,
                    completion_tokens=completion_tokens,
                )
            )

    @staticmethod
    def _avg_optional(vals: List[Optional[float]]) -> float:
        present = [v for v in vals if v is not None]
        if not present:
            return 0.0
        return sum(present) / len(present)

    def snapshot(self, window_seconds: int = 300) -> Dict[str, float]:
        """
        返回最近 window_seconds 秒的统计信息（默认 5 分钟）。

        ``throughput_tokens_per_s`` 使用窗口内各条记录的 ``completion_tokens`` 之和；
        若某条未提供 ``completion_tokens``，则用 ``max(1, completion_len // 4)`` 作为粗估
        token 数以计算吞吐（无真实分词器时的近似）。
        """
        now = time.time()
        with self._lock:
            recent: List[_Record] = [r for r in self._records if now - r.ts <= window_seconds]

        base: Dict[str, float] = {
            "window_seconds": float(window_seconds),
            "request_count": float(len(recent)),
            "avg_duration_ms": 0.0,
            "avg_prompt_len": 0.0,
            "avg_completion_len": 0.0,
            "avg_ttft_ms": 0.0,
            "avg_tpot_ms": 0.0,
            "p50_duration_ms": 0.0,
            "p95_duration_ms": 0.0,
            "throughput_tokens_per_s": 0.0,
            "avg_prefill_ms": 0.0,
            "avg_decode_ms": 0.0,
            "avg_first_token_ms": 0.0,
            "avg_token_window_ms": 0.0,
        }

        count = len(recent)
        if count == 0:
            return base

        total_duration = sum(r.duration_ms for r in recent)
        total_prompt = sum(r.prompt_len for r in recent)
        total_completion = sum(r.completion_len for r in recent)

        durations = [r.duration_ms for r in recent]
        total_completion_tokens = 0
        for r in recent:
            if r.completion_tokens is not None:
                total_completion_tokens += r.completion_tokens
            else:
                # 无真实 tokenizer 时粗估 tok/s
                total_completion_tokens += max(1, r.completion_len // 4)

        total_duration_s = total_duration / 1000.0
        throughput = (total_completion_tokens / total_duration_s) if total_duration_s > 0 else 0.0

        base.update(
            {
                "avg_duration_ms": total_duration / count,
                "avg_prompt_len": total_prompt / count,
                "avg_completion_len": total_completion / count,
                "avg_ttft_ms": self._avg_optional([r.ttft_ms for r in recent]),
                "avg_tpot_ms": self._avg_optional([r.tpot_ms for r in recent]),
                "p50_duration_ms": _percentile_ms(durations, 50.0),
                "p95_duration_ms": _percentile_ms(durations, 95.0),
                "throughput_tokens_per_s": float(throughput),
                "avg_prefill_ms": self._avg_optional([r.prefill_ms for r in recent]),
                "avg_decode_ms": self._avg_optional([r.decode_ms for r in recent]),
                "avg_first_token_ms": self._avg_optional([r.first_token_ms for r in recent]),
                "avg_token_window_ms": self._avg_optional([r.token_window_ms for r in recent]),
            }
        )
        return base


basic_metrics = BasicMetrics()
