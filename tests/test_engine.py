import types

import pytest
import torch

from src.core.config import load_runtime_flags
from src.core.model import MiniVLLMEngine
from src.core.ops import safe_rmsnorm


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


class FollowupUserDummyLLM:
    """模拟模型在回答后继续生成下一轮 User。"""

    def generate(self, prompts, sampling_params):
        class Output:
            def __init__(self, text: str) -> None:
                self.outputs = [types.SimpleNamespace(text=text)]

        return [Output("这是回答内容。\nUser: 继续追问...")]


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


def test_generate_stream_emits_token_chunks_not_chars(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_STREAM_MODE", "token")
    engine = MiniVLLMEngine()
    dummy = DummyLLM()
    _patch_engine_llm(engine, dummy)

    chunks = list(engine.generate("hi", stream=True))
    joined = "".join(chunks)
    assert len(chunks) < len(joined)


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


def test_generate_sets_stop_sequences_to_prevent_self_chat():
    engine = MiniVLLMEngine()
    dummy = DummyLLM()
    _patch_engine_llm(engine, dummy)

    _ = engine.generate("hello", stream=False)

    stop = getattr(dummy._last_params, "stop", None)
    assert stop is not None
    assert "\nUser:" in stop
    assert "\nAssistant:" in stop
    assert "\nSystem:" in stop


def test_generate_strips_followup_user_turn_from_output():
    engine = MiniVLLMEngine()
    _patch_engine_llm(engine, FollowupUserDummyLLM())

    out = engine.generate("hello", stream=False)

    assert out == "这是回答内容。"
    assert "User:" not in out


def test_generate_strips_followup_user_turn_chinese_marker():
    engine = MiniVLLMEngine()

    class ChineseMarkerDummyLLM:
        def generate(self, prompts, sampling_params):
            class Output:
                def __init__(self, text: str) -> None:
                    self.outputs = [types.SimpleNamespace(text=text)]

            return [Output("这是回答内容。\n用户：你再详细说说？")]

    _patch_engine_llm(engine, ChineseMarkerDummyLLM())

    out = engine.generate("hello", stream=False)

    assert out == "这是回答内容。"
    assert "用户：" not in out


def test_generate_strips_followup_user_turn_inline_marker():
    engine = MiniVLLMEngine()

    class InlineUserMarkerDummyLLM:
        def generate(self, prompts, sampling_params):
            class Output:
                def __init__(self, text: str) -> None:
                    self.outputs = [types.SimpleNamespace(text=text)]

            return [Output("LLM是大型语言模型。 User: 请给出一个使用LLM的例子。")]

    _patch_engine_llm(engine, InlineUserMarkerDummyLLM())

    out = engine.generate("50字简述什么是LLM", stream=False)

    assert out == "LLM是大型语言模型。"
    assert "User:" not in out


def test_rmsnorm_triton_flag_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_RMSNORM", "true")
    x = torch.randn(2, 4, dtype=torch.float32)
    w = torch.randn(4, dtype=torch.float32)
    out = safe_rmsnorm(x, w)
    assert out is not None
    assert out.shape == x.shape


def test_rmsnorm_matches_torch_reference(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_RMSNORM", "false")
    x = torch.randn(3, 16, dtype=torch.float32)
    w = torch.randn(16, dtype=torch.float32)
    eps = 1e-6
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
    out = safe_rmsnorm(x, w, eps=eps)
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-4)


def test_decode_attention_flag_exists(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", "true")
    assert load_runtime_flags().enable_decode_attn is True


def test_benchmark_run_id_format():
    import re
    from src.core.benchmarking import build_run_id

    run_id = build_run_id("qwen", "fp16")
    assert run_id.startswith("run-")
    assert len(run_id) > 12
    assert re.match(r"^run-qwen-fp16-\d+$", run_id)
    run_id_pathy = build_run_id("~/models/Qwen/2.5", "bfloat16")
    assert run_id_pathy.startswith("run-") and "bfloat16" in run_id_pathy and "qwen" in run_id_pathy
