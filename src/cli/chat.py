"""
命令行对话入口：
- 支持多轮对话，使用 Conversation 维护上下文
- 支持 --stream 开关控制是否流式打印
- 支持基础参数：--temperature, --max-tokens
"""
from __future__ import annotations

import argparse

from src.core.conversation import Conversation
from src.core.logging import get_logger, setup_logging
from src.core.model import MiniVLLMEngine


logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mini vLLM Qwen CLI chat")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="启用流式输出（逐字符打印模型回复）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="采样温度（默认使用配置中的值）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="最大生成 token 数（默认使用配置中的值）",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="nucleus 采样 top_p（默认使用配置中的值）",
    )
    return parser


def run_cli() -> None:
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    engine = MiniVLLMEngine()
    conv = Conversation()

    print("Mini vLLM Qwen CLI")
    print("输入内容与模型对话，输入 'exit' 或 'quit' 退出。")

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("再见。")
            break

        conv.append_user(user_input)
        prompt = conv.build_prompt()

        print("Assistant: ", end="", flush=True)
        try:
            if args.stream:
                for chunk in engine.generate(
                    prompt,
                    stream=True,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                ):
                    print(chunk, end="", flush=True)
                print()
                # 为对话历史记录完整回答文本
                # 这里简单重新调用一次非流式生成，保证历史中是完整文本
                full_reply = engine.generate(
                    prompt,
                    stream=False,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            else:
                full_reply = engine.generate(
                    prompt,
                    stream=False,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                print(full_reply)

            conv.append_assistant(full_reply)
        except Exception as exc:  # noqa: BLE001
            logger.exception("生成过程中出错: %s", exc)
            print(f"\n[错误] 生成失败：{exc}")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()

