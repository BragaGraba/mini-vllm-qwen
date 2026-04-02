"""M3 decode-attention: vLLM attention_config wiring and standalone grouped-attention kernel."""

import pytest
import torch

from src.core.config import load_runtime_flags
from src.core.ops.decode_attention_runtime import (
    build_llm_extra_kwargs,
    install_decode_attention_patch,
    is_decode_attention_patch_marked,
    uninstall_decode_attention_patch,
)
from src.core.ops.triton_decode_attention import (
    torch_grouped_decode_attention,
    triton_grouped_decode_attention,
)


def test_build_llm_extra_kwargs_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", raising=False)
    assert build_llm_extra_kwargs() == {}


def test_build_llm_extra_kwargs_triton_when_flag_on(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", "true")
    d = build_llm_extra_kwargs()
    assert "attention_config" in d
    cfg = d["attention_config"]
    assert cfg.backend is not None
    assert cfg.backend.name == "TRITON_ATTN"


def test_install_uninstall_patch_mark():
    uninstall_decode_attention_patch()
    assert not is_decode_attention_patch_marked()
    install_decode_attention_patch("test")
    assert is_decode_attention_patch_marked()
    uninstall_decode_attention_patch()
    assert not is_decode_attention_patch_marked()


@pytest.mark.parametrize("b,hq,hkv,s,d", [(1, 8, 2, 64, 128), (2, 4, 4, 32, 64)])
def test_torch_grouped_decode_attention_runs(b, hq, hkv, s, d):
    q = torch.randn(b, hq, d)
    k = torch.randn(b, hkv, s, d)
    v = torch.randn(b, hkv, s, d)
    scale = 1.0 / d**0.5
    out = torch_grouped_decode_attention(q, k, v, scale)
    assert out.shape == (b, hq, d)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_triton_matches_torch_grouped_decode_cuda():
    b, hq, hkv, s, d = 1, 8, 2, 128, 128
    torch.manual_seed(0)
    q = torch.randn(b, hq, d, device="cuda", dtype=torch.float16)
    k = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.float16)
    v = torch.randn(b, hkv, s, d, device="cuda", dtype=torch.float16)
    scale = 1.0 / d**0.5
    ref = torch_grouped_decode_attention(q, k, v, scale)
    tri = triton_grouped_decode_attention(q, k, v)
    torch.testing.assert_close(tri, ref, rtol=2e-2, atol=2e-2)


def test_llm_engine_passes_attention_config_when_flag(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", "true")
    captured: dict = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import src.core.model as model_mod
    import vllm.entrypoints.llm as vllm_llm_mod

    monkeypatch.setattr(vllm_llm_mod, "LLM", FakeLLM)
    eng = model_mod.MiniVLLMEngine()
    eng._ensure_loaded()
    assert "attention_config" in captured
    assert captured["attention_config"].backend.name == "TRITON_ATTN"


def test_decode_attn_flag_load_runtime_flags(monkeypatch):
    monkeypatch.setenv("MINI_VLLM_ENABLE_TRITON_DECODE_ATTN", "1")
    assert load_runtime_flags().enable_decode_attn is True
