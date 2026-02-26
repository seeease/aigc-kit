"""DeepSeek LLM Provider

环境变量:
    DEEPSEEK_API_KEY: API 密钥（必需）
    DEEPSEEK_MODEL: 模型名（默认 deepseek-chat）
    DEEPSEEK_BASE_URL: API 地址（默认 https://api.deepseek.com）
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

from openai import BadRequestError, OpenAI

from ..base import ChatChunk, ChatResult, LLMProvider
from ._openai_compat import iter_stream, parse_response

logger = logging.getLogger(__name__)


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM Provider（OpenAI 兼容）"""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self._base_url = base_url or os.environ.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        )
        self._model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    @property
    def name(self) -> str:
        return "deepseek"

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
        api_kwargs = self._build_kwargs(
            messages,
            model=model,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            stream=False,
            **kwargs,
        )
        response = self._create_with_fallback(**api_kwargs)
        return self._parse_response(response, model or self._model)

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
        api_kwargs = self._build_kwargs(
            messages,
            model=model,
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

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: int,
        max_completion_tokens: int | None,
        response_format: dict[str, Any] | None,
        stream: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        # DeepSeek temperature 范围 [0.0, 2.0)，百分比映射
        actual_temp = max(0.0, min(temperature / 100 * 2.0, 1.99))
        kwargs: dict[str, Any] = {
            "model": model or self._model,
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
            kwargs["response_format"] = self._downgrade_json_schema(
                response_format,
                messages,
            )

        # 非标准字段通过 extra_body 传递
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

    def _downgrade_json_schema(
        self,
        response_format: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """DeepSeek 不支持 json_schema，降级为 json_object"""
        if response_format.get("type") != "json_schema":
            return response_format

        schema = response_format.get("json_schema", {}).get("schema", {})
        if messages and messages[0].get("role") == "system":
            hint = json.dumps(schema, ensure_ascii=False)
            messages[0]["content"] += f"\n\n请严格按以下 JSON Schema 输出：\n{hint}"
        logger.info("DeepSeek 不支持 json_schema，降级为 json_object")
        return {"type": "json_object"}

    def _create_with_fallback(self, **kwargs: Any) -> Any:
        """调用 API，json_schema 失败时自动降级"""
        try:
            return self._client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            rf = kwargs.get("response_format", {})
            if not isinstance(rf, dict) or rf.get("type") != "json_schema":
                raise
            logger.warning("json_schema 请求失败，降级为 json_object: %s", e)
            schema = rf.get("json_schema", {}).get("schema", {})
            kwargs["response_format"] = {"type": "json_object"}
            msgs = kwargs.get("messages", [])
            if msgs and msgs[0].get("role") == "system":
                hint = json.dumps(schema, ensure_ascii=False)
                msgs[0]["content"] += f"\n\n请严格按以下 JSON Schema 输出：\n{hint}"
            return self._client.chat.completions.create(**kwargs)

    @staticmethod
    def _parse_response(response: Any, model: str) -> ChatResult:
        return parse_response(response, provider="deepseek", model=model)

    def close(self) -> None:
        self._client.close()
