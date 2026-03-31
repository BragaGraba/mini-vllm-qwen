"""
模型加载与生成接口封装：统一 MiniVLLMEngine，从 config 读取模型与并发参数，隐藏 vLLM 细节。
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, List, Sequence, Union

from src.core.config import get_generation_config, get_model_config
from src.core.logging import get_logger

logger = get_logger(__name__)


class MiniVLLMEngine:
    """
    统一的 vLLM 推理引擎封装，隐藏底层 vLLM 的模型加载与生成细节。
    模型名称、设备、并发等从环境变量/配置文件（config）读取。
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_num_seqs: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        dtype: str | None = None,
    ):
        cfg = get_model_config()
        self._model_name = model_name if model_name is not None else cfg.model_name
        self._max_num_seqs = max_num_seqs if max_num_seqs is not None else cfg.max_num_seqs
        self._max_model_len = max_model_len if max_model_len is not None else cfg.max_model_len
        self._gpu_memory_utilization = (
            gpu_memory_utilization if gpu_memory_utilization is not None else cfg.gpu_memory_utilization
        )
        self._dtype = dtype if dtype is not None else cfg.dtype
        self._llm = None
        # 预留的钩子：可在调用前后注入自定义逻辑
        self.hook_preprocess: List[Callable[[str], str]] = []
        self.hook_postprocess: List[Callable[[str], str]] = []

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        logger.info(
            "Loading model: %s (max_model_len=%s, max_num_seqs=%s)",
            self._model_name,
            self._max_model_len,
            self._max_num_seqs,
        )
        self._llm = LLM(
            model=self._model_name,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_memory_utilization,
            max_num_seqs=self._max_num_seqs,
            trust_remote_code=True,
            dtype=self._dtype,
            enforce_eager=True,
        )
        logger.info("Model loaded.")

    @staticmethod
    def _strip_followup_turns(text: str) -> str:
        """
        兜底裁剪：若模型继续生成下一轮角色（User/Assistant/System），
        仅保留当前 assistant 回答正文，防止出现“自问自答”。
        """
        marker_patterns = [
            # 行内角色切换（例如："...。 User: ..."）
            r"(?i)(?<=[\s\u3000])(?:user|assistant|system)\s*[:：]",
            r"(?<=[\s\u3000])(?:用户|助手|系统)\s*[:：]",
            r"(?im)^[ \t]*(?:user|assistant|system)\s*[:：]",
            r"(?m)^[ \t]*(?:用户|助手|系统)\s*[:：]",
            r"(?m)<\|im_start\|>\s*(?:user|assistant|system)",
            r"(?m)<\|start_header_id\|>\s*(?:user|assistant|system)\s*<\|end_header_id\|>",
            r"(?m)^#{2,3}\s*(?:User|Assistant|System)\b",
        ]
        cut_positions: List[int] = []
        for pattern in marker_patterns:
            match = re.search(pattern, text)
            if match is not None:
                cut_positions.append(match.start())

        if not cut_positions:
            return text
        return text[: min(cut_positions)].rstrip()

    def generate(
        self,
        prompt: str,
        *,
        stream: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> Union[str, Iterable[str]]:
        """
        生成文本。支持非流式（返回完整字符串）与流式（返回逐块字符串迭代器）。

        :param prompt: 输入提示。
        :param stream: 是否流式返回；为 True 时返回迭代器，否则返回完整字符串。
        :param max_tokens: 最大生成 token 数。
        :param temperature: 采样温度。
        :param top_p: nucleus 采样 top_p。
        :return: stream=False 时为 str，stream=True 时为 Iterable[str]。
        """
        self._ensure_loaded()
        gen = get_generation_config()
        max_tokens = max_tokens if max_tokens is not None else gen.max_tokens
        temperature = temperature if temperature is not None else gen.temperature
        top_p = top_p if top_p is not None else gen.top_p

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # 遇到下一轮角色前缀时立即停止，避免模型在单次请求里继续“自问自答”。
            stop=[
                "\nUser:",
                "\nAssistant:",
                "\nSystem:",
                "\nuser:",
                "\nassistant:",
                "\nsystem:",
                "\nUser：",
                "\nAssistant：",
                "\nSystem：",
                "\n用户:",
                "\n用户：",
                "<|im_start|>user",
                "<|start_header_id|>user<|end_header_id|>",
            ],
        )

        # 预处理 hook
        for fn in self.hook_preprocess:
            prompt = fn(prompt)

        outputs = self._llm.generate([prompt], sampling_params)
        if not outputs or not outputs[0].outputs:
            text = ""
        else:
            text = outputs[0].outputs[0].text

        for fn in self.hook_postprocess:
            text = fn(text)
        text = self._strip_followup_turns(text)

        if not stream:
            return text

        # 流式：vLLM 离线接口无原生 token 流，按字符逐块 yield 以保持接口一致
        def _stream() -> Iterable[str]:
            for ch in text:
                yield ch

        return _stream()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # 为后续批量推理预留的接口（当前简单串行实现，可视需要改为并行）
    def generate_batch(
        self,
        prompts: Sequence[str],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> List[str]:
        """
        批量生成接口预留：当前实现为对每个 prompt 逐个调用 generate。
        将来如需高效批量推理，可在此处直接调用 vLLM 的批量 API。
        """
        results: List[str] = []
        for p in prompts:
            out = self.generate(
                p,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            results.append(out)
        return results


# 可选别名，与计划中的命名一致
QwenEngine = MiniVLLMEngine
