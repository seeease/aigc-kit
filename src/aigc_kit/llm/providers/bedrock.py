"""Amazon Bedrock LLM Provider (Converse API)

环境变量:
    BEDROCK_MODEL: 模型 ID（默认 anthropic.claude-sonnet-4-20250514-v1:0）
    BEDROCK_REGION: AWS 区域（默认 us-east-1）
    AWS_PROFILE: AWS 配置文件名（可选）
"""

import json
import logging
import os
import uuid
from collections.abc import Iterator
from typing import Any

import boto3

from ..base import ChatChunk, ChatResult, LLMProvider, ToolCall

logger = logging.getLogger(__name__)


class BedrockProvider(LLMProvider):
    """Amazon Bedrock LLM Provider"""

    def __init__(
        self,
        *,
        model: str | None = None,
        region: str | None = None,
        profile: str | None = None,
    ):
        self._model = model or os.environ.get(
            "BEDROCK_MODEL",
            "anthropic.claude-sonnet-4-20250514-v1:0",
        )
        _region = region or os.environ.get("BEDROCK_REGION", "us-east-1")
        _profile = profile or os.environ.get("AWS_PROFILE")
        session_kwargs: dict[str, Any] = {"region_name": _region}
        if _profile:
            session_kwargs["profile_name"] = _profile
        self._client = boto3.Session(**session_kwargs).client("bedrock-runtime")

    @property
    def name(self) -> str:
        return "bedrock"

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
        json_schema_mode = (
            response_format is not None
            and response_format.get("type") == "json_schema"
        )
        api_kwargs = self._build_kwargs(
            messages,
            model=use_model,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            thinking=kwargs.get("thinking"),
            response_format=response_format,
        )
        response = self._client.converse(**api_kwargs)
        result = self._parse_response(response, use_model)

        # json_schema 模式：模型被强制调用合成 tool，将 tool 输入转回 content
        if json_schema_mode and result.tool_calls:
            return ChatResult(
                content=result.tool_calls[0].arguments,
                tool_calls=[],
                provider="bedrock",
                model=use_model,
                usage=result.usage or {},
            )
        return result

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
        json_schema_mode = (
            response_format is not None
            and response_format.get("type") == "json_schema"
        )
        api_kwargs = self._build_kwargs(
            messages,
            model=use_model,
            tools=tools,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            thinking=kwargs.get("thinking"),
            response_format=response_format,
        )
        response = self._client.converse_stream(**api_kwargs)
        stream = response.get("stream", [])

        for event in stream:
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    yield ChatChunk(delta_content=delta["text"])
                if "toolUse" in delta:
                    tu = delta["toolUse"]
                    if json_schema_mode:
                        # json_schema 模式：tool 输入作为 content 流式输出
                        yield ChatChunk(delta_content=tu.get("input", ""))
                    else:
                        yield ChatChunk(tool_call_arguments=tu.get("input", ""))

            if "contentBlockStart" in event:
                start = event["contentBlockStart"].get("start", {})
                if "toolUse" in start and not json_schema_mode:
                    tu = start["toolUse"]
                    yield ChatChunk(
                        tool_call_index=event["contentBlockStart"].get(
                            "contentBlockIndex",
                            0,
                        ),
                        tool_call_id=tu.get("toolUseId", ""),
                        tool_call_name=tu.get("name", ""),
                    )

            if "messageStop" in event:
                reason = event["messageStop"].get("stopReason", "")
                yield ChatChunk(finish_reason=reason)

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _build_kwargs(
        messages: list[dict[str, Any]],
        *,
        model: str,
        tools: list[dict[str, Any]] | None,
        temperature: int,
        max_completion_tokens: int | None,
        thinking: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Bedrock temperature 范围 0.0-1.0，百分比映射
        actual_temp = max(0.0, min(temperature / 100, 1.0))
        system_msgs, conv_msgs = _openai_messages_to_bedrock(messages)
        api_kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": conv_msgs,
            "inferenceConfig": {"temperature": actual_temp},
        }
        if max_completion_tokens:
            api_kwargs["inferenceConfig"]["maxTokens"] = max_completion_tokens
        if system_msgs:
            api_kwargs["system"] = system_msgs
        if tools:
            api_kwargs["toolConfig"] = _openai_tools_to_bedrock(tools)
        if thinking:
            thinking_type = thinking.get("type", "disabled")
            if thinking_type == "enabled":
                budget = thinking.get("budget_tokens", 8192)
                api_kwargs["additionalModelRequestFields"] = {
                    "thinking": {"type": "enabled", "budget_tokens": budget},
                }

        # json_schema response_format → 合成 tool + toolChoice 实现结构化输出
        if response_format and response_format.get("type") == "json_schema":
            json_schema_cfg = response_format.get("json_schema", {})
            schema_name = json_schema_cfg.get("name", "json_output")
            schema = json_schema_cfg.get(
                "schema", {"type": "object", "properties": {}}
            )
            tool_spec = {
                "toolSpec": {
                    "name": schema_name,
                    "description": "Output structured JSON according to the schema",
                    "inputSchema": {"json": schema},
                }
            }
            if "toolConfig" in api_kwargs:
                api_kwargs["toolConfig"]["tools"].append(tool_spec)
            else:
                api_kwargs["toolConfig"] = {"tools": [tool_spec]}
            api_kwargs["toolConfig"]["toolChoice"] = {
                "tool": {"name": schema_name},
            }
            # Bedrock 不允许 thinking + toolChoice 同时使用，移除 thinking
            if "additionalModelRequestFields" in api_kwargs:
                api_kwargs["additionalModelRequestFields"].pop("thinking", None)
                if not api_kwargs["additionalModelRequestFields"]:
                    del api_kwargs["additionalModelRequestFields"]

        return api_kwargs

    @staticmethod
    def _parse_response(response: dict[str, Any], model: str) -> ChatResult:
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        content_text = ""
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if "text" in block:
                content_text += block["text"]
            if "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    ToolCall(
                        call_id=tu.get("toolUseId", ""),
                        name=tu.get("name", ""),
                        arguments=json.dumps(tu.get("input", {}), ensure_ascii=False),
                    )
                )

        usage_data = response.get("usage", {})
        usage = {}
        if usage_data:
            usage = {
                "prompt_tokens": usage_data.get("inputTokens", 0),
                "completion_tokens": usage_data.get("outputTokens", 0),
                "total_tokens": (
                    usage_data.get("inputTokens", 0) + usage_data.get("outputTokens", 0)
                ),
            }
        return ChatResult(
            content=content_text or None,
            tool_calls=tool_calls,
            provider="bedrock",
            model=model,
            usage=usage,
        )


# ── 格式转换工具函数 ──────────────────────────────────────────


def _openai_messages_to_bedrock(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """将 OpenAI 格式 messages 转为 Bedrock Converse 格式"""
    system_msgs: list[dict[str, Any]] = []
    conv_msgs: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_msgs.append({"text": content})
            continue

        if role == "assistant":
            parts: list[dict[str, Any]] = []
            if content:
                parts.append({"text": content})
            for tc in msg.get("tool_calls", []):
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                parts.append(
                    {
                        "toolUse": {
                            "toolUseId": tc.get("id", str(uuid.uuid4())),
                            "name": func.get("name", ""),
                            "input": args,
                        }
                    }
                )
            if parts:
                conv_msgs.append({"role": "assistant", "content": parts})
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            try:
                result_data = json.loads(content)
                result_content = [{"json": result_data}]
            except (json.JSONDecodeError, TypeError):
                result_content = [{"text": content or ""}]
            tool_result = {
                "toolResult": {
                    "toolUseId": tool_call_id,
                    "content": result_content,
                }
            }
            if conv_msgs and conv_msgs[-1].get("role") == "user":
                conv_msgs[-1]["content"].append(tool_result)
            else:
                conv_msgs.append({"role": "user", "content": [tool_result]})
            continue

        # user
        parts = []
        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append({"text": item["text"]})
        if parts:
            conv_msgs.append({"role": "user", "content": parts})

    return system_msgs or None, conv_msgs


def _openai_tools_to_bedrock(tools: list[dict[str, Any]]) -> dict[str, Any]:
    bedrock_tools = []
    for tool in tools:
        func = tool.get("function", {})
        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "inputSchema": {
                        "json": func.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        ),
                    },
                }
            }
        )
    return {"tools": bedrock_tools}
