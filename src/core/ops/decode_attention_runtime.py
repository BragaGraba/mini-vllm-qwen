"""
M3 decode-attention runtime: bridge ``MINI_VLLM_ENABLE_TRITON_DECODE_ATTN`` to vLLM V1 attention backend.

Production path uses vLLM's built-in :class:`TritonAttentionBackend` (``TRITON_ATTN``), which runs
inside the EngineCore worker and covers prefill + decode without monkeypatching internals.

Optional ``install_decode_attention_patch`` / ``uninstall_decode_attention_patch`` are reserved for
future custom ``register_backend`` overrides; they only set a debug mark in this module.
"""
from __future__ import annotations

from typing import Any

from src.core.logging import get_logger

logger = get_logger(__name__)

_PATCH_MARK: str | None = None


def install_decode_attention_patch(reason: str = "M3 hook") -> None:
    """
    Placeholder for future runtime overrides (e.g. ``register_backend(CUSTOM, ...)``).

    vLLM 0.18+ selects backends via ``AttentionConfig``; no monkeypatch is required for TRITON_ATTN.
    """
    global _PATCH_MARK
    _PATCH_MARK = reason
    logger.info(
        "M3 decode-attention hook: %s (TRITON path via LLM attention_config, not monkeypatch).",
        reason,
    )


def uninstall_decode_attention_patch() -> None:
    """Clear patch mark (no vLLM globals modified in default integration)."""
    global _PATCH_MARK
    _PATCH_MARK = None
    logger.info("M3 decode-attention patch mark cleared.")


def is_decode_attention_patch_marked() -> bool:
    return _PATCH_MARK is not None


def build_llm_extra_kwargs(*, enable_triton_decode_attn: bool | None = None) -> dict[str, Any]:
    """
    Extra keyword arguments for ``vllm.LLM(...)`` when M3 decode Triton path is enabled.

    When ``enable_triton_decode_attn`` is None, reads ``load_runtime_flags().enable_decode_attn``.
    """
    if enable_triton_decode_attn is None:
        from src.core.config import load_runtime_flags

        enable_triton_decode_attn = load_runtime_flags().enable_decode_attn

    if not enable_triton_decode_attn:
        return {}

    from vllm.config import AttentionConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    return {
        "attention_config": AttentionConfig(backend=AttentionBackendEnum.TRITON_ATTN),
    }
