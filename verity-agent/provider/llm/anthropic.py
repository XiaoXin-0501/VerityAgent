"""Anthropic Messages API adapter。"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

from config.provider import ChatProviderConfig
from provider.enums import (
    ChatFinishReason,
    ModelRole,
    ProviderErrorCode,
    ToolChoiceMode,
)
from exception.provider import ProviderError
from api.network import HttpProviderMixin
from provider.llm.base import ChatModel, ChatProvider
from provider.llm.types import (
    ChatModelRequest,
    ChatModelResponse,
    ChatStreamEnd,
    ChatStreamError,
    ChatStreamEvent,
    ChatStreamStart,
    ChatTextDelta,
    ChatThinkingDelta,
    ChatToolCallDelta,
    ChatToolCallEnd,
    ChatToolCallStart,
    ChatUsage,
    ChatUsageEvent,
    ModelContentPart,
    ModelFilePart,
    ModelImagePart,
    ModelMessage,
    ModelTextPart,
    ModelThinkingPart,
    ModelToolCall,
)

__all__ = ["AnthropicChatModel", "AnthropicChatProvider"]

_STOP_MAP = {
    "end_turn": ChatFinishReason.STOP,
    "stop_sequence": ChatFinishReason.STOP,
    "stop": ChatFinishReason.STOP,
    "max_tokens": ChatFinishReason.LENGTH,
    "tool_use": ChatFinishReason.TOOL_CALLS,
    "refusal": ChatFinishReason.CONTENT_FILTER,
    "error": ChatFinishReason.ERROR,
    "cancelled": ChatFinishReason.CANCELLED,
}


def _finish_reason(value: Any) -> ChatFinishReason:
    return _STOP_MAP.get(str(value), ChatFinishReason.STOP)


def _usage(payload: Mapping[str, Any] | None) -> ChatUsage:
    data = payload or {}
    return ChatUsage(
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        total_tokens=int(data.get("input_tokens") or 0)
        + int(data.get("output_tokens") or 0),
        cached_input_tokens=int(data.get("cache_read_input_tokens") or 0),
    )


def _image_block(part: ModelImagePart) -> dict[str, Any]:
    if part.data:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type or "image/png",
                "data": base64.b64encode(part.data).decode("ascii"),
            },
        }
    if part.url:
        return {"type": "image", "source": {"type": "url", "url": part.url}}
    raise ProviderError(
        ProviderErrorCode.INVALID_REQUEST, "图片 part 缺少 url/data"
    )


def _blocks(parts: list[ModelContentPart]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ModelTextPart):
            blocks.append({"type": "text", "text": part.text})
        elif isinstance(part, ModelImagePart):
            blocks.append(_image_block(part))
        elif isinstance(part, ModelFilePart):
            if part.text:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[file {part.filename or ''}]\n{part.text}",
                    }
                )
            elif part.url:
                blocks.append({"type": "text", "text": part.url})
        elif isinstance(part, ModelThinkingPart):
            if part.redacted:
                continue
            block: dict[str, Any] = {"type": "thinking", "thinking": part.text}
            if part.signature:
                block["signature"] = part.signature
            blocks.append(block)
    return blocks


class AnthropicChatModel(ChatModel):
    def __init__(self, spec, provider: "AnthropicChatProvider") -> None:
        super().__init__(spec)
        self.provider = provider

    def _messages(self, messages: list[ModelMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        result: list[dict[str, Any]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role is ModelRole.SYSTEM:
                system_parts.extend(
                    part.text
                    for part in message.content
                    if isinstance(part, ModelTextPart)
                )
                index += 1
                continue
            if message.role is ModelRole.TOOL:
                # 连续 tool 结果合并成一个 user message
                blocks: list[dict[str, Any]] = []
                while index < len(messages) and messages[index].role is ModelRole.TOOL:
                    tool = messages[index]
                    text = "".join(
                        part.text
                        for part in tool.content
                        if isinstance(part, ModelTextPart)
                    )
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool.tool_call_id,
                            "content": text or json.dumps(_blocks(tool.content)),
                        }
                    )
                    index += 1
                result.append({"role": "user", "content": blocks})
                continue

            content = _blocks(message.content)
            item: dict[str, Any] = {
                "role": "assistant" if message.role is ModelRole.ASSISTANT else "user",
                "content": content,
            }
            if message.role is ModelRole.ASSISTANT:
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.call_id,
                            "name": call.name,
                            "input": call.arguments or {},
                        }
                    )
            result.append(item)
            index += 1
        return ("\n\n".join(system_parts) if system_parts else None), result

    def _tool_choice(self, request: ChatModelRequest) -> dict[str, Any] | None:
        if not request.tools:
            return None
        choice = request.tool_choice
        if choice.mode is ToolChoiceMode.NONE:
            return {"type": "none"}
        if choice.mode is ToolChoiceMode.REQUIRED:
            return {"type": "any"}
        if choice.mode is ToolChoiceMode.TOOL:
            return {"type": "tool", "name": choice.tool_name}
        return {"type": "auto"}

    def _payload(self, request: ChatModelRequest, *, stream: bool) -> dict[str, Any]:
        system, messages = self._messages(request.messages)
        if request.prefix and (not messages or messages[-1]["role"] != "assistant"):
            raise ProviderError(
                ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                "Anthropic prefix completion 要求最后一条是 assistant 消息",
                provider=self.provider.config.name,
                model=self.spec.model,
            )
        options = request.options
        max_tokens = self.option(
            "max_output_tokens", options.max_output_tokens
        ) or self.spec.chat_capabilities.max_output_tokens
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "max_tokens": max_tokens,
            "messages": messages,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in request.tools
            ]
            choice = self._tool_choice(request)
            if choice:
                payload["tool_choice"] = choice
        temperature = self.option("temperature", options.temperature)
        if temperature is not None:
            payload["temperature"] = temperature
        top_p = self.option("top_p", options.top_p)
        if top_p is not None:
            payload["top_p"] = top_p
        stop = options.stop or self._defaults.get("stop") or []
        if stop:
            payload["stop_sequences"] = stop
        thinking_budget = self.option(
            "thinking_budget", options.thinking_budget
        )
        if thinking_budget is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        provider_options = dict(self._defaults.get("provider_options") or {})
        provider_options.update(options.provider_options)
        if provider_options:
            payload.update(provider_options)
        # Anthropic 不接收 OpenAI 的 response_format；结构化输出由 LLM 提示词约束
        return payload

    def _response(self, data: Mapping[str, Any]) -> ChatModelResponse:
        parts: list[ModelContentPart] = []
        calls: list[ModelToolCall] = []
        for block in data.get("content") or []:
            if not isinstance(block, Mapping):
                continue
            kind = block.get("type")
            if kind == "text":
                parts.append(ModelTextPart(text=str(block.get("text") or "")))
            elif kind == "thinking":
                parts.append(
                    ModelThinkingPart(
                        text=str(block.get("thinking") or ""),
                        signature=block.get("signature"),
                    )
                )
            elif kind == "redacted_thinking":
                parts.append(ModelThinkingPart(text="", redacted=True))
            elif kind == "tool_use":
                calls.append(
                    ModelToolCall(
                        call_id=str(block.get("id") or ""),
                        name=str(block.get("name") or ""),
                        arguments=block.get("input") or {},
                    )
                )
        return ChatModelResponse(
            message=ModelMessage(
                role=ModelRole.ASSISTANT, content=parts, tool_calls=calls
            ),
            finish_reason=_finish_reason(data.get("stop_reason")),
            usage=_usage(data.get("usage")),
            response_id=data.get("id"),
        )

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        data = await self.provider.request_json(
            "POST",
            "messages",
            payload=self._payload(request, stream=False),
            model=self.spec.model,
        )
        return self._response(data)

    async def stream(self, request: ChatModelRequest) -> AsyncIterator[ChatStreamEvent]:
        payload = self._payload(request, stream=True)
        yield ChatStreamStart(model=self.spec.model)
        blocks: dict[int, dict[str, Any]] = {}
        tool_json: dict[int, str] = {}
        usage: ChatUsage | None = None
        finish_reason = ChatFinishReason.STOP
        response_id: str | None = None
        try:
            async for event_name, data_text in self.provider.stream_sse(
                "messages", payload=payload, model=self.spec.model
            ):
                try:
                    data = json.loads(data_text)
                except ValueError:
                    continue
                kind = data.get("type") or event_name
                if kind == "error":
                    error = data.get("error") or {}
                    raise ProviderError(
                        ProviderErrorCode.UPSTREAM,
                        str(error.get("message") or "Anthropic stream error"),
                        provider=self.provider.config.name,
                        model=self.spec.model,
                        provider_code=error.get("type"),
                    )
                if kind == "message_start":
                    message = data.get("message") or {}
                    response_id = message.get("id")
                    usage = _usage(message.get("usage"))
                elif kind == "content_block_start":
                    index = int(data.get("index") or 0)
                    block = data.get("content_block") or {}
                    blocks[index] = block
                    if block.get("type") == "tool_use":
                        yield ChatToolCallStart(
                            call_id=str(block.get("id") or ""),
                            name=str(block.get("name") or ""),
                            index=index,
                        )
                elif kind == "content_block_delta":
                    index = int(data.get("index") or 0)
                    delta = data.get("delta") or {}
                    block = blocks.get(index, {})
                    if delta.get("type") == "text_delta":
                        yield ChatTextDelta(text=str(delta.get("text") or ""), index=index)
                    elif delta.get("type") == "thinking_delta":
                        yield ChatThinkingDelta(
                            text=str(delta.get("thinking") or ""),
                            signature=delta.get("signature"),
                            index=index,
                        )
                    elif delta.get("type") == "input_json_delta":
                        call_id = str(block.get("id") or "")
                        fragment = str(delta.get("partial_json") or "")
                        tool_json[index] = tool_json.get(index, "") + fragment
                        yield ChatToolCallDelta(
                            call_id=call_id,
                            arguments_delta=fragment,
                            index=index,
                        )
                elif kind == "content_block_stop":
                    index = int(data.get("index") or 0)
                    block = blocks.get(index, {})
                    if block.get("type") == "tool_use":
                        raw = tool_json.get(index)
                        arguments = None
                        if raw:
                            try:
                                parsed = json.loads(raw)
                                arguments = parsed if isinstance(parsed, dict) else None
                            except ValueError:
                                arguments = None
                        yield ChatToolCallEnd(
                            call_id=str(block.get("id") or ""),
                            arguments=arguments,
                            raw_arguments=raw,
                            index=index,
                        )
                elif kind == "message_delta":
                    delta = data.get("delta") or {}
                    if delta.get("stop_reason"):
                        finish_reason = _finish_reason(delta.get("stop_reason"))
                    if data.get("usage"):
                        current = usage or ChatUsage()
                        usage = ChatUsage(
                            input_tokens=current.input_tokens,
                            output_tokens=int(data["usage"].get("output_tokens") or 0),
                            total_tokens=current.input_tokens
                            + int(data["usage"].get("output_tokens") or 0),
                            cached_input_tokens=current.cached_input_tokens,
                        )
                        yield ChatUsageEvent(usage=usage)
                elif kind == "message_stop":
                    break
        except ProviderError as exc:
            yield ChatStreamError(error=exc)
            return
        yield ChatStreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
        )


class AnthropicChatProvider(ChatProvider, HttpProviderMixin):
    model_class = AnthropicChatModel
    default_capabilities = {
        "thinking": True,
        "vision": True,
        "file_input": True,
        "prefix_completion": True,
        "prompt_caching": True,
    }

    def __init__(self, config: ChatProviderConfig) -> None:
        super().__init__(config)
        self._init_http()
        self._handles: dict[str, ChatModel] = {}

    def auth_headers(self) -> dict[str, str]:
        api_key = self.config.resolve_api_key()
        version = self.config.metadata.get("anthropic_version", "2023-06-01")
        headers = {"anthropic-version": str(version)}
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    async def open(self) -> None:
        await super().open()
        await self.open_http()

    async def close(self) -> None:
        await self.close_http()
        await super().close()

    def get_model(self, model_id: str) -> ChatModel:
        model = self._handles.get(model_id)
        if model is None:
            model = self.model_class(self.get_spec(model_id), self)
            self._handles[model_id] = model
        return model
