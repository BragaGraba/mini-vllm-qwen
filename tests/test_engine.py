import types

import pytest

from src.core.model import MiniVLLMEngine


class DummyLLM:
    """不实际加载大模型的假 LLM，仅用于测试接口逻辑。"""

    def __init__(self) -> None:
        self._last_prompts = None
        self._last_params = None

    def generate(self, prompts, sampling_params):
        self._last_prompts = prompts
        self._last_params = sampling_params

        class Output:
            def __init__(self, text: str) -> None:
                self.outputs = [types.SimpleNamespace(text=text)]

        # 回显 + 简单标记，方便断言
        return [Output(f"echo: {prompts[0]}")]


def _patch_engine_llm(engine: MiniVLLMEngine, dummy: DummyLLM) -> None:
    engine._llm = dummy  # type: ignore[attr-defined]


def test_generate_non_stream_uses_hooks_and_returns_text(monkeypatch):
    engine = MiniVLLMEngine()
    dummy = DummyLLM()
    _patch_engine_llm(engine, dummy)

    # 通过 hook_preprocess / hook_postprocess 验证调用链
    engine.hook_preprocess.append(lambda p: p + " [pre]")
    engine.hook_postprocess.append(lambda t: t + " [post]")

    out = engine.generate("hello", stream=False, max_tokens=8, temperature=0.5, top_p=0.9)

    assert isinstance(out, str)
    assert out.endswith("[post]")
    # DummyLLM 接收到的 prompt 应该带有 [pre]
    assert dummy._last_prompts[0].endswith("[pre]")


def test_generate_stream_iterates_chunks(monkeypatch):
    engine = MiniVLLMEngine()
    dummy = DummyLLM()
    _patch_engine_llm(engine, dummy)

    chunks = list(engine.generate("hi", stream=True))
    # 默认 DummyLLM 输出 "echo: hi"，流式时按字符返回
    assert "".join(chunks).startswith("echo:")


def test_generate_batch_calls_generate(monkeypatch):
    engine = MiniVLLMEngine()
    dummy = DummyLLM()
    _patch_engine_llm(engine, dummy)

    # 使用 monkeypatch 监控 generate 调用次数
    calls = {}

    real_generate = engine.generate

    def wrapped_generate(prompt: str, **kwargs):
        calls["count"] = calls.get("count", 0) + 1
        return real_generate(prompt, **kwargs)

    monkeypatch.setattr(engine, "generate", wrapped_generate)

    prompts = ["p1", "p2", "p3"]
    outputs = engine.generate_batch(prompts)

    assert len(outputs) == len(prompts)
    assert calls["count"] == len(prompts)

