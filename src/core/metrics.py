"""
轻量级监控：记录最近一段时间内的调用统计信息。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List


@dataclass
class _Record:
    ts: float
    duration_ms: float
    prompt_len: int
    completion_len: int


class BasicMetrics:
    """
    仅在内存中维护最近 N 条生成调用记录，用于 /metrics/basic。
    不依赖外部存储，重启即清空。
    """

    def __init__(self, max_records: int = 200) -> None:
        self._records: Deque[_Record] = deque(maxlen=max_records)
        self._lock = threading.Lock()

    def record(self, *, duration_ms: float, prompt_len: int, completion_len: int) -> None:
        with self._lock:
            self._records.append(
                _Record(
                    ts=time.time(),
                    duration_ms=duration_ms,
                    prompt_len=prompt_len,
                    completion_len=completion_len,
                )
            )

    def snapshot(self, window_seconds: int = 300) -> Dict[str, float]:
        """
        返回最近 window_seconds 秒的统计信息（默认 5 分钟）。
        """
        now = time.time()
        with self._lock:
            recent: List[_Record] = [r for r in self._records if now - r.ts <= window_seconds]

        count = len(recent)
        if count == 0:
            return {
                "window_seconds": float(window_seconds),
                "request_count": 0.0,
                "avg_duration_ms": 0.0,
                "avg_prompt_len": 0.0,
                "avg_completion_len": 0.0,
            }

        total_duration = sum(r.duration_ms for r in recent)
        total_prompt = sum(r.prompt_len for r in recent)
        total_completion = sum(r.completion_len for r in recent)

        return {
            "window_seconds": float(window_seconds),
            "request_count": float(count),
            "avg_duration_ms": total_duration / count,
            "avg_prompt_len": total_prompt / count,
            "avg_completion_len": total_completion / count,
        }


basic_metrics = BasicMetrics()

