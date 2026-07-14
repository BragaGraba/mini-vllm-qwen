# mini-vllm-qwen

基于 vLLM 的 Qwen2.5-3B-Instruct 轻量封装：CLI / REST API / Web UI。

## 环境要求

- **操作系统**：Linux / WSL2
- **Python**：推荐 `Python 3.10` 及以上
- **GPU / CUDA**：NVIDIA GPU（示例：RTX 3060 12GB，驱动 536.19，CUDA 12.2）
- **依赖**：`pip install -r requirements.txt`（含 FastAPI / Uvicorn；vLLM 需自行按环境安装）

## 安装与运行

1. 创建并激活虚拟环境（可选但推荐）：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. 安装依赖：

   ```bash
   pip install -r requirements.txt
   ```

3. （可选）在项目根目录创建 `.env`：

   ```bash
   MINI_VLLM_MODEL=~/models/Qwen2.5-3B-Instruct
   MINI_VLLM_MAX_NUM_SEQS=2
   MINI_VLLM_MAX_MODEL_LEN=4096
   MINI_VLLM_GPU_MEMORY_UTILIZATION=0.8
   MINI_VLLM_WARMUP_ON_STARTUP=true
   MINI_VLLM_LOG_LEVEL=INFO
   MINI_VLLM_API_PORT=8000
   ```

   `MINI_VLLM_WARMUP_ON_STARTUP` 默认 `true`；设为 `false` 可关闭启动预热。

## 启动方式

- **API / Web**：`uvicorn src.api.server:app --host 0.0.0.0 --port 8000`
- **CLI**：`python -m src.cli.chat`

参考（裸 vLLM serve）：

```bash
vllm serve ~/models/Qwen2.5-3B-Instruct \
  --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.8 --max-num-seqs 2 \
  --swap-space 4 --port 8000
```

## REST API 示例

- **健康检查**：`curl http://localhost:8000/health`

- **非流式对话**

  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-3b",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "用一句话介绍一下你自己。"}
      ],
      "stream": false,
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.9
    }'
  ```

- **流式对话（SSE）**

  ```bash
  curl -N -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "qwen2.5-3b",
      "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "流式帮我写一段鼓励自己的话。"}
      ],
      "stream": true,
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.9
    }'
  ```

- **基础监控**：`curl "http://localhost:8000/metrics/basic?window_seconds=60"`

OpenAPI：`http://localhost:8000/docs`。Web UI：服务根路径 `/`。

## 可选优化开关

默认关闭。需要时通过环境变量开启：

| 变量 | 作用 |
|------|------|
| `MINI_VLLM_ENABLE_TRITON_RMSNORM=true` | Triton RMSNorm（失败回退 PyTorch） |
| `MINI_VLLM_ENABLE_TRITON_DECODE_ATTN=true` | vLLM `AttentionConfig(backend=TRITON_ATTN)` |

流式切分：`MINI_VLLM_STREAM_MODE=char|token`（默认 `char`；仍为完整生成后再切分，非内核级逐 token）。
