"""Optional fused / Triton-backed ops with safe fallbacks."""

from src.core.ops.decode_attention_runtime import (
    build_llm_extra_kwargs,
    install_decode_attention_patch,
    uninstall_decode_attention_patch,
)
from src.core.ops.triton_decode_attention import (
    safe_grouped_decode_attention,
    torch_grouped_decode_attention,
    triton_grouped_decode_attention,
)
from src.core.ops.triton_rmsnorm import (
    safe_rmsnorm,
    torch_rmsnorm_fallback,
    torch_rmsnorm_reference,
)

__all__ = [
    "build_llm_extra_kwargs",
    "install_decode_attention_patch",
    "uninstall_decode_attention_patch",
    "safe_grouped_decode_attention",
    "torch_grouped_decode_attention",
    "triton_grouped_decode_attention",
    "safe_rmsnorm",
    "torch_rmsnorm_fallback",
    "torch_rmsnorm_reference",
]
