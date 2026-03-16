"""
轻量对话状态管理：Conversation / ChatSession，用于维护最近 N 轮对话并构建 prompt。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal


Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class Conversation:
    """
    最小会话结构：
    - 维护一组消息（Message）
    - 支持追加用户/助手消息
    - 支持仅保留最近 N 轮对话
    - 提供 build_prompt() 构建给模型的输入文本
    """

    messages: List[Message] = field(default_factory=list)
    max_rounds: int = 4
    system_prompt: str | None = None

    def append_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))
        self._trim()

    def append_assistant(self, content: str) -> None:
        self.messages.append(Message(role="assistant", content=content))
        self._trim()

    def set_system(self, content: str) -> None:
        """设置/更新 system 提示词。"""
        self.system_prompt = content

    def _trim(self) -> None:
        """
        仅保留最近 max_rounds 轮 user+assistant 对话。
        system 提示词单独存放，不在 messages 里裁剪。
        """
        if self.max_rounds <= 0:
            return

        # 统计从后往前的 user/assistant 配对轮次
        rounds = 0
        idx = len(self.messages) - 1
        # 简单近似：遇到 user 记一轮，最多保留 max_rounds 轮
        while idx >= 0 and rounds < self.max_rounds:
            if self.messages[idx].role == "user":
                rounds += 1
            idx -= 1

        # idx 停在需要裁剪的位置（包含 idx），只保留之后的消息
        if idx > 0:
            self.messages = self.messages[idx + 1 :]

    def build_prompt(self) -> str:
        """
        构建用于喂给 Qwen/vLLM 的文本 prompt。
        首版采用简单拼接形式：

        - 可选 system 提示在最前面
        - 每条消息前加前缀：User:/Assistant:/System:
        """
        parts: List[str] = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")

        for msg in self.messages:
            if msg.role == "user":
                prefix = "User"
            elif msg.role == "assistant":
                prefix = "Assistant"
            else:
                prefix = "System"
            parts.append(f"{prefix}: {msg.content}")

        return "\n".join(parts) + "\nAssistant:"


ChatSession = Conversation

