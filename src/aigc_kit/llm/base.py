"""LLM Provider 抽象接口

每个 provider 从环境变量读取默认配置，调用时按需覆盖。
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """工具调用"""

    call_id: str
    name: str
    arguments: str  # JSON 字符串


@dataclass
class ChatResult:
    """Chat 响应结果"""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


@dataclass
class ChatChunk:
    """流式响应片段"""

    delta_content: str = ""
    tool_call_index: int | None = None
    tool_call_id: str = ""
    tool_call_name: str = ""
    tool_call_arguments: str = ""
    finish_reason: str | None = None


class LLMProvider(ABC):
    """LLM Provider 抽象基类

    子类实现时应从环境变量读取默认配置（API key、model 等），
    构造函数参数作为可选覆盖。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider 名称"""

    @property
    @abstractmethod
    def default_model(self) -> str:
        """当前使用的模型名"""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: int = 70,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步 Chat Completions

        Args:
            temperature: 温度百分比 0-100，由各 provider 映射到实际范围
            thinking: 思考配置，如 {"type": "enabled"} 或 {"type": "disabled"}，
                      各 provider 自行映射到对应 API 格式
        """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: int = 70,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        """流式 Chat Completions

        Args:
            temperature: 温度百分比 0-100，由各 provider 映射到实际范围
            thinking: 思考配置，各 provider 自行映射到对应 API 格式
        """

    def close(self) -> None:
        """释放资源（可选覆盖）"""

    def _log_request(
        self,
        method: str,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        temperature: int,
        max_completion_tokens: int | None,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """统一的请求日志（DEBUG 级别）"""
        logger.debug(
            "=== %s.%s ===\nmodel: %s\ntemperature: %d\n"
            "max_completion_tokens: %s\ntools: %d\nresponse_format: %s\n"
            "extra: %s\nmessages:\n%s",
            self.name,
            method,
            model or self.default_model,
            temperature,
            max_completion_tokens,
            len(tools) if tools else 0,
            response_format.get("type") if response_format else None,
            kwargs or None,
            json.dumps(messages, ensure_ascii=False, indent=2),
        )
