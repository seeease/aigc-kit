"""OpenAI 兼容 provider 的公共工具函数"""

from collections.abc import Iterator
from typing import Any

from ..base import ChatChunk, ChatResult, ToolCall


def parse_response(response: Any, *, provider: str, model: str) -> ChatResult:
    """解析 OpenAI 兼容 API 的同步响应"""
    msg = response.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        tool_calls = [
            ToolCall(
                call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments,
            )
            for tc in msg.tool_calls
        ]
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return ChatResult(
        content=msg.content,
        tool_calls=tool_calls,
        provider=provider,
        model=model,
        usage=usage,
    )


def iter_stream(stream: Any) -> Iterator[ChatChunk]:
    """解析 OpenAI 兼容 API 的流式响应"""
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish = chunk.choices[0].finish_reason

        if delta.content:
            yield ChatChunk(delta_content=delta.content, finish_reason=finish)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                yield ChatChunk(
                    tool_call_index=tc_delta.index,
                    tool_call_id=tc_delta.id or "",
                    tool_call_name=(
                        tc_delta.function.name
                        if tc_delta.function and tc_delta.function.name
                        else ""
                    ),
                    tool_call_arguments=(
                        tc_delta.function.arguments
                        if tc_delta.function and tc_delta.function.arguments
                        else ""
                    ),
                    finish_reason=finish,
                )

        if finish and not delta.content and not delta.tool_calls:
            yield ChatChunk(finish_reason=finish)
