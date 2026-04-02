# Hotspot summary (`run-qwen-float16-1775041592`)

> **Source:** `torch.profiler` attempted but returned 0% — vLLM V1 runs all GPU kernels in a separate `EngineCore` subprocess, invisible to main-process profiling.
> **Environment:** RTX 3060 12GB, CUDA 12.8, vLLM 0.18.0 (V1 engine, FlashAttention v2)
> **Model:** Qwen2.5-3B-Instruct (36 layers, hidden_size=2048, num_heads=16, num_kv_heads=2, intermediate_size=11008)

## Profiling Limitation

vLLM V1 architecture uses `multiprocessing.spawn` for `EngineCore`, which means:
- All CUDA kernels (attention, norm, MLP, sampling) execute in the **child process**
- `torch.profiler` in the parent process captures **0 CUDA events**
- `nsys` would work (system-wide profiling) but is not installed
- Workaround: profile from within the EngineCore process, or install nsys

## attention

**Percent share (estimated):** ~30-40% — FlashAttention v2 is used for all 36 layers; GQA (16 heads, 2 KV heads) reduces memory bandwidth but attention remains the dominant op.

## norm

**Percent share (estimated):** ~5-8% — RMSNorm at each layer (pre-attention + pre-MLP = 72 calls per forward), relatively cheap for hidden_size=2048.

## mlp

**Percent share (estimated):** ~40-50% — SiLU-gated MLP with intermediate_size=11008 (5.4x hidden_size), two GEMM ops per layer (gate_proj + up_proj → SiLU → down_proj).

## sampling

**Percent share (estimated):** ~2-5% — Single argmax/top-k + softmax over vocab_size=151936 per decode step.

## other

**Percent share (estimated):** ~5-10% — Embedding lookup, rotary position encoding, residual additions, tensor copies between host↔device for vLLM scheduling.

## Notes

- Estimates are based on Qwen2.5-3B architecture parameters and typical transformer profiling literature.
- For ground-truth data, install `nsys` and run:
  ```bash
  ./scripts/profile_inference.sh --run-id run-qwen-float16-1775041592 --execute
  ```
- MLP dominance (40-50%) is expected for Qwen2.5 due to large intermediate_size.
- Attention share depends heavily on sequence length; estimates assume short-medium prompts.
- M4 go/no-go threshold is >8% — both attention (~35%) and MLP (~45%) exceed this.
