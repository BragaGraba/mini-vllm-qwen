"""Optional fused / Triton-backed ops with safe fallbacks."""

from src.core.ops.triton_rmsnorm import (
    safe_rmsnorm,
    torch_rmsnorm_fallback,
    torch_rmsnorm_reference,
)

__all__ = [
    "safe_rmsnorm",
    "torch_rmsnorm_fallback",
    "torch_rmsnorm_reference",
]
