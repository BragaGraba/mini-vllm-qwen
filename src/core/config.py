"""
基础配置结构：模型名称、最大并发、默认生成参数等，从 .env 或环境变量读取。
"""
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env_str(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_path(key: str, default: str = "") -> str:
    raw = _env_str(key, default)
    return os.path.expanduser(raw)


def _env_int(key: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = _env_str(key, str(default)).lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass
class ModelConfig:
    """模型与推理相关配置。"""

    model_name: str = field(default_factory=lambda: _env_path("MINI_VLLM_MODEL", "~/models/Qwen/Qwen2.5-3B-Instruct"))
    """模型名称或路径。"""
    max_num_seqs: int = field(default_factory=lambda: _env_int("MINI_VLLM_MAX_NUM_SEQS", 1))
    """最大并发序列数。"""
    max_model_len: int = field(default_factory=lambda: _env_int("MINI_VLLM_MAX_MODEL_LEN", 4096))
    """最大模型上下文长度。"""
    gpu_memory_utilization: float = field(default_factory=lambda: _env_float("MINI_VLLM_GPU_MEMORY_UTILIZATION", 0.8))
    """GPU 显存利用率。"""
    dtype: str = field(default_factory=lambda: _env_str("MINI_VLLM_DTYPE", "float16"))


@dataclass
class GenerationConfig:
    """默认生成参数。"""

    max_tokens: int = field(default_factory=lambda: _env_int("MINI_VLLM_DEFAULT_MAX_TOKENS", 512))
    temperature: float = field(default_factory=lambda: _env_float("MINI_VLLM_DEFAULT_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("MINI_VLLM_DEFAULT_TOP_P", 0.9))


@dataclass
class AppConfig:
    """应用级配置（API 端口等）。"""

    api_host: str = field(default_factory=lambda: _env_str("MINI_VLLM_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: _env_int("MINI_VLLM_API_PORT", 8000))
    log_level: str = field(default_factory=lambda: _env_str("MINI_VLLM_LOG_LEVEL", "INFO").upper())
    warmup_on_startup: bool = field(default_factory=lambda: _env_bool("MINI_VLLM_WARMUP_ON_STARTUP", True))


# 单例式全局配置（按需可改为显式注入）
_model_config: Optional[ModelConfig] = None
_generation_config: Optional[GenerationConfig] = None
_app_config: Optional[AppConfig] = None


def get_model_config() -> ModelConfig:
    global _model_config
    if _model_config is None:
        _model_config = ModelConfig()
    return _model_config


def get_generation_config() -> GenerationConfig:
    global _generation_config
    if _generation_config is None:
        _generation_config = GenerationConfig()
    return _generation_config


def get_app_config() -> AppConfig:
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


def get_benchmark_output_dir() -> str:
    """Directory for benchmark CSV/metadata; empty when unset."""
    return _env_path("MINI_VLLM_BENCHMARK_OUT", "")


def get_skip_concurrency_env() -> tuple[int | None, str]:
    """If MINI_VLLM_SKIP_CONCURRENCY is set to a positive int, that concurrency is marked skipped."""
    n = _env_int("MINI_VLLM_SKIP_CONCURRENCY", 0)
    if n <= 0:
        return None, ""
    reason = _env_str(
        "MINI_VLLM_SKIP_CONCURRENCY_REASON",
        "skipped via MINI_VLLM_SKIP_CONCURRENCY",
    )
    return n, reason


def get_stream_mode() -> Literal["char", "token"]:
    """
    Streaming chunking mode from MINI_VLLM_STREAM_MODE: ``char`` (default) or ``token``.
    Unknown values fall back to ``char``.
    """
    raw = _env_str("MINI_VLLM_STREAM_MODE", "char").lower()
    if raw == "token":
        return "token"
    return "char"
