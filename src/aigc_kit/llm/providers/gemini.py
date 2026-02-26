"""Google Gemini LLM Provider

环境变量:
    GEMINI_MODEL: 模型名（默认 gemini-2.5-flash）
    GOOGLE_CLOUD_PROJECT: GCP 项目 ID（Vertex AI 模式）
    GOOGLE_CLOUD_LOCATION: GCP 区域（Vertex AI 模式）
"""

import json
import logging
import os
from collections.abc import Iterator
from typing import Any

from google import genai
from google.genai import types

from ..base import ChatChunk, ChatResult, LLMProvider, ToolCall

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM Provider"""

    def __init__(
        self,
        *,
        model: str | None = None,
        project: str | None = None,
        location: str | None = None,
    ):
        self._model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        client_kwargs: dict[str, Any] = {}
        _project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        _location = location or os.environ.get("GOOGLE_CLOUD_LOCATION")
        if _project:
            client_kwargs["project"] = _project
        if _location:
            client_kwargs["location"] = _location
        self._client = genai.Client(**client_kwargs)

    @property
    def name(self) -> str:
        return "gemini"

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
        system_text, contents = _openai_messages_to_gemini(messages)
        config = self._build_config(
            system_text=system_text,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            thinking=kwargs.get("thinking"),
        )
        use_model = model or self._model
        response = self._client.models.generate_content(
            model=use_model,
            contents=contents,
            config=config,
        )
        return self._parse_response(response, use_model)

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
        system_text, contents = _openai_messages_to_gemini(messages)
        config = self._build_config(
            system_text=system_text,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format=response_format,
            thinking=kwargs.get("thinking"),
        )
        use_model = model or self._model
        stream = self._client.models.generate_content_stream(
            model=use_model,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            finish = candidate.finish_reason
            for part in candidate.content.parts:
                if part.text:
                    yield ChatChunk(
                        delta_content=part.text,
                        finish_reason=str(finish) if finish else None,
                    )
                if part.function_call:
                    fc = part.function_call
                    yield ChatChunk(
                        tool_call_index=0,
                        tool_call_id=fc.id or "",
                        tool_call_name=fc.name or "",
                        tool_call_arguments=json.dumps(
                            dict(fc.args) if fc.args else {},
                            ensure_ascii=False,
                        ),
                        finish_reason=str(finish) if finish else None,
                    )

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _build_config(
        *,
        system_text: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: int,
        max_completion_tokens: int | None,
        response_format: dict[str, Any] | None,
        thinking: dict[str, Any] | None = None,
    ) -> types.GenerateContentConfig:
        # Gemini temperature 范围 [0.0, 2.0)，百分比映射
        actual_temp = max(0.0, min(temperature / 100 * 2.0, 1.99))
        config_kwargs: dict[str, Any] = {"temperature": actual_temp}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if max_completion_tokens:
            config_kwargs["max_output_tokens"] = max_completion_tokens
        if tools:
            config_kwargs["tools"] = _openai_tools_to_gemini(tools)
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
        if response_format:
            rf_type = response_format.get("type", "")
            if rf_type in ("json_schema", "json_object"):
                config_kwargs["response_mime_type"] = "application/json"
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    config_kwargs["response_json_schema"] = schema
        if thinking:
            thinking_type = thinking.get("type", "disabled")
            if thinking_type == "enabled":
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking.get("budget_tokens", 8192),
                )
            elif thinking_type == "disabled":
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=0,
                )
        return types.GenerateContentConfig(**config_kwargs)

    @staticmethod
    def _parse_response(response: Any, model: str) -> ChatResult:
        candidate = response.candidates[0]
        content_text = ""
        tool_calls: list[ToolCall] = []
        for part in candidate.content.parts:
            if part.text:
                content_text += part.text
            if part.function_call:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        call_id=fc.id or f"call_{fc.name}",
                        name=fc.name,
                        arguments=json.dumps(
                            dict(fc.args) if fc.args else {},
                            ensure_ascii=False,
                        ),
                    )
                )
        usage = {}
        if response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": um.prompt_token_count or 0,
                "completion_tokens": um.candidates_token_count or 0,
                "total_tokens": um.total_token_count or 0,
            }
        return ChatResult(
            content=content_text or None,
            tool_calls=tool_calls,
            provider="gemini",
            model=model,
            usage=usage,
        )


# ── 格式转换工具函数 ──────────────────────────────────────────


def _openai_messages_to_gemini(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[types.Content]]:
    """将 OpenAI 格式 messages 转为 Gemini 格式"""
    system_text = None
    contents: list[types.Content] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_text = content
            continue

        if role == "assistant":
            parts: list[types.Part] = []
            if content:
                parts.append(types.Part.from_text(text=content))
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                args = json.loads(func.get("arguments", "{}"))
                parts.append(
                    types.Part.from_function_call(
                        name=func.get("name", ""),
                        args=args,
                    )
                )
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            func_name = _find_tool_name(messages, tool_call_id)
            try:
                result_data = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                result_data = {"result": content}
            contents.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=func_name,
                            response=result_data,
                        )
                    ],
                )
            )
            continue

        # user
        parts = []
        if isinstance(content, str) and content:
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
        if parts:
            contents.append(types.Content(role="user", parts=parts))

    return system_text, contents


def _find_tool_name(messages: list[dict[str, Any]], tool_call_id: str) -> str:
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []):
            if tc.get("id") == tool_call_id:
                return tc.get("function", {}).get("name", "")
    return "unknown"


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[types.Tool]:
    declarations = []
    for tool in tools:
        func = tool.get("function", {})
        declarations.append(
            types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters_json_schema=func.get("parameters", {}),
            )
        )
    return [types.Tool(function_declarations=declarations)]
