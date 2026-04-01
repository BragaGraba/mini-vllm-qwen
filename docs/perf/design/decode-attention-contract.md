# M3 Decode-Attention (Triton) — Implementation Contract

**Spec:** [2026-04-01 vLLM v1 inference optimization design](../../superpowers/specs/2026-04-01-vllm-v1-inference-optimization-design.md) (§5.1 P0, M3).

## Objectives

- Replace or augment the default decode-phase attention path with a Triton kernel that improves memory access (e.g., paged KV, small-batch decode) while meeting TPOT/throughput targets from the spec.
- Preserve numerical quality within agreed tolerances (`rtol`/`atol` vs reference) and avoid regressions under long-context and multi-sequence concurrency.

## Expected interface (handoff)

**Inputs (conceptual):** query/key/value (or fused QKV) tensors, attention metadata (sequence lengths, block tables / slot mapping as required by the integration layer), model constants (`head_dim`, scaling), and dtype/device consistent with the existing engine.

**Outputs:** attention output tensor(s) matching the shape and semantics of the current CUDA path for supported configurations; unsupported shapes must fail closed into the existing fallback.

**Integration boundary:** changes stay behind the repo’s operator/replacement layer and `MiniVLLMEngine.generate()` remains the stable entrypoint; kernel selection is gated by configuration (see below), not by callers of `generate()`.

## Rollback

- Set **`MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=0`** (or unset) to force the legacy path. Runtime code should read flags via `load_runtime_flags()` from `src.core.config` so behavior tracks the process environment.
- On compile failure, shape mismatch, or guarded error, implementations must fall back to the default attention implementation without crashing the server.
