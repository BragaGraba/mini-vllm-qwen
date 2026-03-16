from fastapi.testclient import TestClient

from src.api.server import app


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

