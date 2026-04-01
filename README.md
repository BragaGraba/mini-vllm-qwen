# 模型启动命令
vllm serve ~/models/Qwen2.5-3B-Instruct   --dtype float16   --max-model-len 4096   --gpu-memory-utilization 0.8   --max-num-seqs 2   --swap-space 4   --port 8000

## 环境要求

- **操作系统**：Linux / WSL2（当前开发环境为 WSL2 + Ubuntu 22.04）
- **Python**：推荐 `Python 3.10` 及以上
- **GPU / CUDA**：
  - NVIDIA GPU（示例环境：GeForce RTX 3060 12GB）
  - NVIDIA 驱动（示例：536.19）
  - CUDA 版本（示例：12.2）
- **vLLM**：在本项目虚拟环境中通过 `pip install -r requirements.txt` 安装（`vllm>=0.4.0`）

## 环境确认（阶段 0 / 任务 0.1）

- Python 版本：`Python 3.10.12`（通过 `python3 -V` 检查）
- vLLM Python 包：当前环境下 `pip show vllm` 未找到，后续需要在本项目虚拟环境中安装。
- GPU/驱动：
  - 使用 `nvidia-smi` 检查，已检测到 `NVIDIA GeForce RTX 3060`
  - 驱动版本 `536.19`，CUDA 版本 `12.2`

> 说明：上面的 `vllm serve` 启动命令是已有环境中的参考命令，后续项目将基于相同模型（Qwen2.5-3B-Instruct）封装 CLI / REST API / Web UI。

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

3. （可选）在项目根目录创建 `.env`，配置模型路径、端口、默认参数等，例如：

   ```bash
   MINI_VLLM_MODEL=~/models/Qwen2.5-3B-Instruct
   MINI_VLLM_MAX_NUM_SEQS=2
   MINI_VLLM_MAX_MODEL_LEN=4096
   MINI_VLLM_GPU_MEMORY_UTILIZATION=0.8
   MINI_VLLM_WARMUP_ON_STARTUP=true
   MINI_VLLM_LOG_LEVEL=INFO
   MINI_VLLM_API_PORT=8000
   ```

   说明：`MINI_VLLM_WARMUP_ON_STARTUP` 默认 `true`，服务启动时会做一次轻量模型预热；设置为 `false` 可关闭预热。

## 运行方式约定（阶段 0 / 任务 0.3）

- **API 服务**：使用 Uvicorn 启动 FastAPI 应用（实现见阶段 3）  
  `uvicorn src.api.server:app --host 0.0.0.0 --port 8000`

- **CLI 对话**：使用模块方式启动（实现见阶段 2）  
  `python -m src.cli.chat`

- **日志**：项目内统一使用 Python 标准库 `logging`，格式与初始化在 `src/core/logging.py`（阶段 1 实现）。

## REST API 示例（阶段 3 / 任务 3.1–3.4）

- **健康检查**

  ```bash
  curl http://localhost:8000/health
  ```

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

- **基础监控（GET /metrics/basic）**

  ```bash
  curl "http://localhost:8000/metrics/basic"
  curl "http://localhost:8000/metrics/basic?window_seconds=60"
  ```

  返回 JSON：`window_seconds`、`request_count`、`avg_duration_ms`、`avg_prompt_len`、`avg_completion_len`。Web 页顶部监控栏会定期调用该接口并展示。

> 说明：完整 OpenAPI/Swagger 文档可在服务启动后访问 `http://localhost:8000/docs`。

## Profiling

Use `scripts/profile_inference.sh` to capture Nsight Systems (`nsys`) profiles of the benchmark driver and write a normalized hotspot summary under `docs/perf/profiles/`. See `docs/perf/profiles/README.md` for options, GPU requirements, and how it pairs with baseline CSVs.