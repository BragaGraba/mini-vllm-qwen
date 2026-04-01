"""
RMSNorm: PyTorch reference and optional Triton backend behind MINI_VLLM_ENABLE_TRITON_RMSNORM.
"""
from __future__ import annotations

import torch

from src.core.config import get_triton_rmsnorm_enabled

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _rmsnorm_row_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_row,
        n_cols,
        eps_f,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_SIZE)
        mask = offs < n_cols
        base = row * stride_row
        x_row = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        w_row = tl.load(w_ptr + offs, mask=mask, other=0.0)
        xf = x_row.to(tl.float32)
        sq = xf * xf
        var = tl.sum(sq, axis=0) / n_cols.to(tl.float32) + eps_f
        inv_rms = tl.rsqrt(var)
        y = x_row * inv_rms * w_row
        tl.store(out_ptr + base + offs, y, mask=mask)

    _TRITON_IMPORT_OK = True
except ImportError:
    _rmsnorm_row_kernel = None  # type: ignore[misc, assignment]
    _TRITON_IMPORT_OK = False


def torch_rmsnorm_fallback(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm over the last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def _triton_rmsnorm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Triton path: 2D contiguous CUDA tensors, last dim <= 8192."""
    if not _TRITON_IMPORT_OK or _rmsnorm_row_kernel is None:
        raise ImportError("triton not available")

    if x.device.type != "cuda" or not x.is_cuda:
        raise RuntimeError("Triton RMSNorm pilot expects CUDA tensors")
    if x.dim() != 2:
        raise RuntimeError("Triton RMSNorm pilot expects 2D tensors")
    m, n = x.shape
    if n > 8192:
        raise RuntimeError("last dim too large for pilot kernel")

    x_c = x.contiguous()
    w_c = weight.contiguous().to(device=x.device, dtype=x.dtype)
    out = torch.empty_like(x_c)

    stride_xm, stride_xn = x_c.stride()
    if stride_xn != 1:
        raise RuntimeError("expected contiguous last dimension")

    block = int(triton.next_power_of_2(n))
    if block < 128:
        block = 128

    grid = (m,)
    _rmsnorm_row_kernel[grid](
        x_c,
        w_c,
        out,
        x_c.stride(0),
        n,
        float(eps),
        BLOCK_SIZE=block,
    )
    return out


def safe_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm with optional Triton backend when ``get_triton_rmsnorm_enabled()`` is True.
    On any failure (missing Triton, compile, runtime), falls back to PyTorch.
    """
    if not get_triton_rmsnorm_enabled():
        return torch_rmsnorm_fallback(x, weight, eps)
    try:
        return _triton_rmsnorm_forward(x, weight, eps)
    except Exception:
        return torch_rmsnorm_fallback(x, weight, eps)


torch_rmsnorm_reference = torch_rmsnorm_fallback
