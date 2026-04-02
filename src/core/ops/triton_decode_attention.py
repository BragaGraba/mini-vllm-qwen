"""
Repo-local grouped decode attention (one Q step vs cached K/V), with optional Triton implementation.

Production M3 wires ``MINI_VLLM_ENABLE_TRITON_DECODE_ATTN`` to vLLM's ``TRITON_ATTN`` backend
(see ``decode_attention_runtime.build_llm_extra_kwargs``). This file provides:

- ``torch_grouped_decode_attention``: reference for tests
- ``triton_grouped_decode_attention``: Triton path (CUDA, fp16/bf16/fp32) when Triton is available

Shapes: ``q`` [B, Hq, D], ``k`` / ``v`` [B, Hkv, S, D], ``Hq % Hkv == 0``. Sequence length must be ``<= max_seq_len``.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_OK = True
except ImportError:
    _TRITON_IMPORT_OK = False
    triton = None  # type: ignore[misc, assignment]
    tl = None  # type: ignore[misc, assignment]


def torch_grouped_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """PyTorch grouped attention over full KV length (decode cache)."""
    B, Hq, D = q.shape
    _, Hkv, S, _ = k.shape
    if Hq % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    g = Hq // Hkv
    qg = q.view(B, Hkv, g, D)
    scores = torch.einsum("bhgd,bhsd->bhgs", qg.float(), k.float()) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhgs,bhsd->bhgd", attn, v.float()).reshape(B, Hq, D)
    return out.to(dtype=q.dtype)


if _TRITON_IMPORT_OK:

    @triton.jit
    def _grouped_decode_attn_kernel(
        Q_ptr,
        K_ptr,
        V_ptr,
        Out_ptr,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_od,
        num_kv_heads,
        group_size,
        seq_len,
        scale,
        HEAD_DIM: tl.constexpr,
        BLOCK_S: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
    ):
        hq = tl.program_id(0)
        b = tl.program_id(1)
        hkv = hq // group_size

        offs_d = tl.arange(0, HEAD_DIM)
        mask_d = offs_d < HEAD_DIM
        q = tl.load(
            Q_ptr + b * stride_qb + hq * stride_qh + offs_d * stride_qd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)

        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

        base_k = b * stride_kb + hkv * stride_kh
        base_v = b * stride_vb + hkv * stride_vh

        for block_idx in range(MAX_BLOCKS):
            start = block_idx * BLOCK_S
            offs_s = start + tl.arange(0, BLOCK_S)
            mask_s = offs_s < seq_len

            k_ptrs = K_ptr + base_k + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
            k_block = tl.load(k_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)

            q_col = tl.reshape(q, (HEAD_DIM, 1))
            scores = tl.dot(k_block, q_col)
            scores = tl.reshape(scores, (BLOCK_S,))
            scores = scores * scale
            scores = tl.where(mask_s, scores, float("-inf"))

            m_block = tl.max(scores)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(scores - m_new)
            p = tl.where(mask_s, p, 0.0)
            l_new = alpha * l_i + tl.sum(p)

            v_ptrs = V_ptr + base_v + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
            v_block = tl.load(v_ptrs, mask=mask_s[:, None], other=0.0).to(tl.float32)

            p_row = tl.reshape(p, (1, BLOCK_S))
            contrib = tl.dot(p_row, v_block)
            contrib = tl.reshape(contrib, (HEAD_DIM,))

            acc = acc * alpha + contrib
            m_i = m_new
            l_i = l_new

        out = acc / l_i
        tl.store(
            Out_ptr + b * stride_ob + hq * stride_oh + offs_d * stride_od,
            out,
            mask=mask_d,
        )


def _supports_triton_cuda(q: torch.Tensor, k: torch.Tensor, head_dim: int, seq_len: int) -> bool:
    if not _TRITON_IMPORT_OK:
        return False
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if head_dim not in (32, 64, 128):
        return False
    if seq_len > 8192:
        return False
    return True


def triton_grouped_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped decode attention via Triton (CUDA). Falls back to PyTorch if unsupported.

    Uses a fixed upper bound of 256 * 32 = 8192 sequence positions in the kernel loop
    (masked by ``seq_len``).
    """
    B, Hq, D = q.shape
    _, Hkv, S, _ = k.shape
    if Hq % Hkv != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    scale = 1.0 / math.sqrt(D)

    if not _supports_triton_cuda(q, k, D, S):
        return torch_grouped_decode_attention(q, k, v, scale)

    BLOCK_S = 32
    MAX_BLOCKS = 256

    out = torch.empty(B, Hq, D, device=q.device, dtype=torch.float32)
    grid = (Hq, B)
    _grouped_decode_attn_kernel[grid](
        q,
        k,
        v,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        Hkv,
        Hq // Hkv,
        S,
        scale,
        HEAD_DIM=D,
        BLOCK_S=BLOCK_S,
        MAX_BLOCKS=MAX_BLOCKS,
    )
    return out.to(dtype=q.dtype)


def safe_grouped_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    prefer_triton: bool = True,
) -> torch.Tensor:
    """Try Triton on CUDA when ``prefer_triton``; otherwise PyTorch reference."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    if not prefer_triton:
        return torch_grouped_decode_attention(q, k, v, scale)
    try:
        return triton_grouped_decode_attention(q, k, v)
    except Exception:
        return torch_grouped_decode_attention(q, k, v, scale)
