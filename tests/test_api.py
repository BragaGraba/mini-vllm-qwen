from fastapi.testclient import TestClient

from src.api.server import app
from src.core.metrics import basic_metrics


client = TestClient(app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "model_name" in data


def test_chat_completions_non_stream(monkeypatch):
    # 避免测试时真正加载大模型：用简单假实现替换 MiniVLLMEngine.generate
    from src.api import server as server_module

    def fake_generate(prompt: str, **kwargs) -> str:
        return "test-reply"

    monkeypatch.setattr(server_module.state.engine, "generate", fake_generate)

    payload = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "你好"},
        ],
        "stream": False,
    }

    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "test-reply"


def test_metrics_snapshot_includes_ttft_and_tpot():
    basic_metrics.clear()
    basic_metrics.record(
        duration_ms=100.0,
        prompt_len=10,
        completion_len=8,
        ttft_ms=40.0,
        tpot_ms=15.0,
    )
    data = basic_metrics.snapshot(window_seconds=300)
    assert "avg_ttft_ms" in data
    assert "avg_tpot_ms" in data
    assert data["avg_ttft_ms"] == 40.0
    assert data["avg_tpot_ms"] == 15.0


def test_metrics_snapshot_includes_percentiles_and_throughput():
    basic_metrics.clear()
    basic_metrics.record(
        duration_ms=100.0,
        prompt_len=5,
        completion_len=8,
        completion_tokens=2,
    )
    basic_metrics.record(
        duration_ms=200.0,
        prompt_len=5,
        completion_len=8,
        completion_tokens=2,
    )
    basic_metrics.record(
        duration_ms=300.0,
        prompt_len=5,
        completion_len=8,
        completion_tokens=2,
    )
    data = basic_metrics.snapshot(window_seconds=300)
    assert "p50_duration_ms" in data
    assert "p95_duration_ms" in data
    assert "throughput_tokens_per_s" in data
    assert data["p50_duration_ms"] == 200.0
    # Linear rank 1.9 between 200 and 300 -> 200 + 0.9 * (300 - 200) == 290
    assert data["p95_duration_ms"] == 290.0
    # total_completion_tokens=6, total_duration_s=0.6 -> 10 tok/s
    assert abs(data["throughput_tokens_per_s"] - 10.0) < 1e-6


def test_metrics_snapshot_includes_stage_keys():
    basic_metrics.clear()
    basic_metrics.record(
        duration_ms=50.0,
        prompt_len=3,
        completion_len=4,
        prefill_ms=10.0,
        decode_ms=40.0,
    )
    data = basic_metrics.snapshot(window_seconds=300)
    assert "avg_prefill_ms" in data
    assert "avg_decode_ms" in data
    assert data["avg_prefill_ms"] == 10.0
    assert data["avg_decode_ms"] == 40.0


def test_metrics_snapshot_includes_token_timing_windows():
    basic_metrics.clear()
    basic_metrics.record(
        duration_ms=80.0,
        prompt_len=2,
        completion_len=4,
        first_token_ms=25.0,
        token_window_ms=12.5,
    )
    data = basic_metrics.snapshot(window_seconds=300)
    assert "avg_first_token_ms" in data
    assert "avg_token_window_ms" in data
    assert data["avg_first_token_ms"] == 25.0
    assert data["avg_token_window_ms"] == 12.5

