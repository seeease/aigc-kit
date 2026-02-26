"""火山引擎豆包 LLM Provider

环境变量:
    VOLCENGINE_API_KEY: API 密钥（必需）
    VOLCENGINE_MODEL: 模型名/推理接入点（默认 doubao-1.5-pro-32k-250115）
    VOLCENGINE_BASE_URL: API 地址（默认 https://ark.cn-beijing.volces.com/api/v3）
"""

import logging
import os
from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from ..base import ChatChunk, ChatResult, LLMProvider
from ._openai_compat import iter_stream, parse_response

logger = logging.getLogger(__name__)


class VolcEngineProvider(LLMProvider):
    """火山引擎豆包 LLM Provider（OpenAI 兼容）"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or os.environ["VOLCENGINE_API_KEY"]
        self._base_url = base_url or os.environ.get(
            "VOLCENGINE_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        self._model = model or os.environ.get(
            "VOLCENGINE_MODEL",
            "doubao-1.5-pro-32k-250115",
        )
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def name(self) -> str:
        return "volcengine"

    @property
    def default_model(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: int = 70,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._log_request(
            "chat",
            messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )
        use_model = model or self._model
        api_kwargs = self._build_kwargs(
            messages,
            model=use_model,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            stream=False,
            **kwargs,
        )
        response = self._client.chat.completions.create(**api_kwargs)
        return parse_response(response, provider="volcengine", model=use_model)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: int = 70,
        max_completion_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatChunk]:
        self._log_request(
            "chat_stream",
            messages,
            model=model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            tools=tools,
            response_format=response_format,
            **kwargs,
        )
        use_model = model or self._model
        api_kwargs = self._build_kwargs(
            messages,
            model=use_model,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            stream=True,
            **kwargs,
        )
        stream = self._client.chat.completions.create(**api_kwargs)
        yield from iter_stream(stream)

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _build_kwargs(
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: int,
        max_completion_tokens: int | None,
        response_format: dict[str, Any] | None,
        stream: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        # 豆包 temperature 范围 0.0-1.0，百分比映射
        actual_temp = max(0.0, min(temperature / 100, 1.0))
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": actual_temp,
            "stream": stream,
        }
        if max_completion_tokens:
            kwargs["max_completion_tokens"] = max_completion_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if response_format:
            kwargs["response_format"] = response_format

        sdk_params = {
            "response_format",
            "max_completion_tokens",
            "tools",
            "tool_choice",
        }
        extra_body = {k: v for k, v in extra.items() if k not in sdk_params}
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def close(self) -> None:
        self._client.close()
