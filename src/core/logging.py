"""
统一日志初始化：时间 + 级别 + 模块 + 消息。
"""
import logging
import sys

from src.core.config import get_app_config


def setup_logging(
    level: str | None = None,
    format_string: str | None = None,
) -> None:
    """初始化根 logger，统一格式与级别。"""
    app = get_app_config()
    log_level = (level or app.log_level).upper()
    if format_string is None:
        format_string = "%(asctime)s %(levelname)s %(name)s %(message)s"

    level_value = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(
        level=level_value,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """获取已配置格式的 logger。若尚未调用 setup_logging，会先执行默认初始化。"""
    root = logging.getLogger()
    if not root.handlers:
        setup_logging()
    return logging.getLogger(name)
