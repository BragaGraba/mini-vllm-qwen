from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional
from pathlib import Path
import time
import json

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.core.config import get_app_config
from src.core.conversation import Conversation
from src.core.logging import get_logger, setup_logging
from src.core.metrics import basic_metrics
from src.core.model import MiniVLLMEngine


setup_logging()
logger = get_logger(__name__)

app = FastAPI(title="Mini vLLM Qwen API", version="0.1.0")


class MiniVLLMState:
    """简单的全局状态，封装引擎与共享会话。"""

    def __init__(self) -> None:
        self.engine = MiniVLLMEngine()
        self.conversation = Conversation()


state = MiniVLLMState()


def _completion_tokens_estimate(completion_len: int) -> int:
    """无真实 tokenizer 时用字符长度粗估 token 数（与 metrics 吞吐启发式一致）。"""
    return max(1, completion_len // 4)


@app.on_event("startup")
def warmup_model() -> None:
    """
    在服务启动阶段完成一次轻量预热，避免首个真实请求承担模型冷启动开销。
    """
    cfg = get_app_config()
    if not cfg.warmup_on_startup:
        logger.info("Model warmup skipped (MINI_VLLM_WARMUP_ON_STARTUP=false).")
        return

    logger.info("Model warmup started...")
    start_ts = time.perf_counter()
    # 使用极小生成参数触发模型加载和首次编译缓存。
    state.engine.generate(
        "你好",
        stream=False,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
    )
    duration_ms = (time.perf_counter() - start_ts) * 1000.0
    logger.info("Model warmup finished in %.2f ms.", duration_ms)


@app.get("/health")
def health() -> Dict[str, Any]:
    """健康检查接口。"""
    cfg = get_app_config()
    return {
        "status": "ok",
        "model_loaded": state.engine.is_loaded,
        "model_name": state.engine.model_name,
        "api_host": cfg.api_host,
        "api_port": cfg.api_port,
    }


def _build_prompt(messages: List[Dict[str, str]]) -> str:
    """
    从 OpenAI 风格 messages 构建/更新 Conversation，并返回 prompt。
    请求体示例：
    {
      "model": "qwen",
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."}
      ],
      "stream": false,
      "max_tokens": 256,
      "temperature": 0.7,
      "top_p": 0.9
    }
    """
    conv = state.conversation

    # 重置会话：根据请求中的 messages 重新构造简单历史
    conv.messages.clear()
    conv.system_prompt = None

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            conv.set_system(content)
        elif role == "assistant":
            conv.append_assistant(content)
        else:
            conv.append_user(content)

    return conv.build_prompt()


def _build_response_object(
    content: str,
    *,
    stream: bool,
    model: str,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """构造一个近似 OpenAI 的响应 JSON 对象（非流式使用）。"""
    usage: Dict[str, int] = {}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage["completion_tokens"] = completion_tokens
    if usage:
        usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

    return {
        "id": "mini-vllm-qwen-chat",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage or None,
        "stream": stream,
    }


@app.post("/v1/chat/completions")
def chat_completions(body: Dict[str, Any]) -> Any:
    """
    兼容 OpenAI 风格的聊天接口：
    - 非流式：默认，返回 JSON
    - 流式：body.stream = true 时，返回 StreamingResponse（SSE：每条 data: JSON 后以空行结束）
    """
    try:
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=400, detail="`messages` must be a non-empty list")

        stream: bool = bool(body.get("stream", False))
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")
        top_p = body.get("top_p")
        model_name = body.get("model") or state.engine.model_name

        prompt = _build_prompt(messages)
        prompt_len = len(prompt)

        if not stream:
            # 非流式
            start_ts = time.perf_counter()
            content = state.engine.generate(
                prompt,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            duration_ms = (time.perf_counter() - start_ts) * 1000.0
            # 更新会话历史与监控
            state.conversation.append_assistant(content)
            comp_len = len(content)
            ct_est = _completion_tokens_estimate(comp_len)
            # 非流式无首 token 边界：TTFT/首 token 时间近似为整段耗时（见 limitation）
            basic_metrics.record(
                duration_ms=duration_ms,
                prompt_len=prompt_len,
                completion_len=comp_len,
                ttft_ms=duration_ms,
                tpot_ms=duration_ms / ct_est,
                prefill_ms=0.0,
                decode_ms=duration_ms,
                first_token_ms=duration_ms,
                token_window_ms=duration_ms / ct_est,
                completion_tokens=ct_est,
            )
            resp_obj = _build_response_object(
                content,
                stream=False,
                model=model_name,
            )
            return JSONResponse(resp_obj)

        # 流式接口：SSE（data: <json>\\n\\n）
        async def event_stream() -> AsyncGenerator[bytes, None]:
            try:
                buffer: List[str] = []
                start_ts = time.perf_counter()
                first_chunk_ms: float | None = None
                for chunk in state.engine.generate(
                    prompt,
                    stream=True,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ):
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.perf_counter() - start_ts) * 1000.0
                    buffer.append(chunk)
                    data = {
                        "id": "mini-vllm-qwen-chat",
                        "object": "chat.completion.chunk",
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": chunk},
                                "finish_reason": None,
                            }
                        ],
                    }
                    payload = json.dumps(data, ensure_ascii=False)
                    yield f"data: {payload}\n\n".encode("utf-8")

                full_text = "".join(buffer)
                duration_ms = (time.perf_counter() - start_ts) * 1000.0
                state.conversation.append_assistant(full_text)
                comp_len = len(full_text)
                ct_est = _completion_tokens_estimate(comp_len)
                ttft = first_chunk_ms if first_chunk_ms is not None else duration_ms
                decode_span = max(0.0, duration_ms - ttft)
                tpot_ms = decode_span / ct_est if ct_est else duration_ms
                basic_metrics.record(
                    duration_ms=duration_ms,
                    prompt_len=prompt_len,
                    completion_len=comp_len,
                    ttft_ms=ttft,
                    tpot_ms=tpot_ms,
                    prefill_ms=0.0,
                    decode_ms=duration_ms,
                    first_token_ms=ttft,
                    token_window_ms=decode_span / ct_est if ct_est else None,
                    completion_tokens=ct_est,
                )
                # 结束标记
                done_message = {
                    "id": "mini-vllm-qwen-chat",
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                }
                done_payload = json.dumps(done_message, ensure_ascii=False)
                yield f"data: {done_payload}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Streaming generation failed: %s", exc)
                error_payload = {"error": {"message": str(exc)}}
                error_json = json.dumps(error_payload, ensure_ascii=False)
                yield f"data: {error_json}\n\n".encode("utf-8")

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_completions failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"internal error: {exc}") from exc


WEB_INDEX_PATH = (Path(__file__).resolve().parent.parent / "web" / "index.html").resolve()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """返回 Web 对话界面。"""
    return FileResponse(WEB_INDEX_PATH)


@app.get("/metrics/basic")
def metrics_basic(window_seconds: int = Query(300, ge=60, le=3600, description="统计时间窗口（秒）")) -> Dict[str, Any]:
    """
    返回最近一段时间（默认 5 分钟）的统计（内存 deque，重启清空），包含：
    - request_count, avg_duration_ms, avg_prompt_len, avg_completion_len
    - avg_ttft_ms, avg_tpot_ms, p50_duration_ms, p95_duration_ms
    - throughput_tokens_per_s（窗口内估计输出 token 量 / 总请求耗时，近似吞吐）
    - avg_prefill_ms, avg_decode_ms（占位/代理字段）
    - avg_first_token_ms, avg_token_window_ms
    """
    return basic_metrics.snapshot(window_seconds=window_seconds)


@app.get("/v1/chat/stream")
def chat_stream(q: str = Query(..., description="用户输入的最新一条消息")) -> StreamingResponse:
    """
    为 Web 前端提供的简化流式接口（使用 EventSource 调用）：
    - 仅使用本次 user 输入 q 与固定 system 提示构造 prompt
    - 不复用会话历史（前端仍会展示本地历史）
    """

    sys_prompt = "You are a helpful Chinese-speaking assistant."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": q},
    ]
    prompt = _build_prompt(messages)
    prompt_len = len(prompt)
    model_name = state.engine.model_name

    async def event_stream() -> AsyncGenerator[bytes, None]:
        try:
            buffer: List[str] = []
            start_ts = time.perf_counter()
            first_chunk_ms: float | None = None
            for chunk in state.engine.generate(prompt, stream=True):
                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - start_ts) * 1000.0
                buffer.append(chunk)
                data = {
                    "id": "mini-vllm-qwen-chat",
                    "object": "chat.completion.chunk",
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                payload = json.dumps(data, ensure_ascii=False)
                yield f"data: {payload}\n\n".encode("utf-8")

            full_text = "".join(buffer)
            duration_ms = (time.perf_counter() - start_ts) * 1000.0
            comp_len = len(full_text)
            ct_est = _completion_tokens_estimate(comp_len)
            ttft = first_chunk_ms if first_chunk_ms is not None else duration_ms
            decode_span = max(0.0, duration_ms - ttft)
            tpot_ms = decode_span / ct_est if ct_est else duration_ms
            basic_metrics.record(
                duration_ms=duration_ms,
                prompt_len=prompt_len,
                completion_len=comp_len,
                ttft_ms=ttft,
                tpot_ms=tpot_ms,
                prefill_ms=0.0,
                decode_ms=duration_ms,
                first_token_ms=ttft,
                token_window_ms=decode_span / ct_est if ct_est else None,
                completion_tokens=ct_est,
            )
            done_message = {
                "id": "mini-vllm-qwen-chat",
                "object": "chat.completion.chunk",
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": "stop",
                    }
                ],
            }
            done_payload = json.dumps(done_message, ensure_ascii=False)
            yield f"data: {done_payload}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("chat_stream failed: %s", exc)
            error_payload = {"error": {"message": str(exc)}}
            error_json = json.dumps(error_payload, ensure_ascii=False)
            yield f"data: {error_json}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

