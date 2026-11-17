"""Anthropic Messages API adapter。"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import anthropic

from config.provider import ChatProviderConfig
from provider.enums import (
    ChatFinishReason,
    ModelRole,
    ToolChoiceMode,
)
from exception.provider import ProviderError, ProviderErrorCode
from provider.llm.base import ChatModel, ChatProvider
from provider.sdk import (
    build_anthropic_client,
    sdk_dump,
    sdk_error,
    split_sdk_kwargs,
)
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

_ANTHROPIC_KWARGS = frozenset(
    {
        "model",
        "max_tokens",
        "messages",
        "stream",
        "system",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "stop_sequences",
        "thinking",
        "metadata",
    }
)

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
    """把 Anthropic stop_reason 归一化为统一结束原因。"""
    return _STOP_MAP.get(str(value), ChatFinishReason.STOP)


def _usage(payload: Mapping[str, Any] | None) -> ChatUsage:
    """从 usage 字典构造统一 token 用量（含缓存命中数）。"""
    data = payload or {}
    return ChatUsage(
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        total_tokens=int(data.get("input_tokens") or 0)
        + int(data.get("output_tokens") or 0),
        cached_input_tokens=int(data.get("cache_read_input_tokens") or 0),
    )


def _image_block(part: ModelImagePart) -> dict[str, Any]:
    """把图片 part 转成 Anthropic base64 或 URL image block。"""
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
    """把统一 content part 列表转换为 Anthropic content blocks。"""
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
        """绑定模型规格与所属 Provider。"""
        super().__init__(spec)
        self.provider = provider

    def _messages(self, messages: list[ModelMessage]) -> tuple[str | None, list[dict[str, Any]]]:
        """拆分 system 文本并转换消息；连续 tool 结果合并为一条 user 消息。"""
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
        """把统一工具选择策略转换为 Anthropic tool_choice。"""
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
        """构造 Messages API 请求体；prefix completion 要求末条为 assistant。"""
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
        thinking_enabled = self.spec.metadata.get("thinking_enabled")
        thinking_budget = self.option(
            "thinking_budget", options.thinking_budget
        )
        if thinking_enabled is not False and thinking_budget is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
        if thinking_enabled is False:
            # Anthropic 默认不启用 thinking；显式移除预算，避免请求仍带
            # thinking 字段。若模型需要厂商级关闭参数，由配置注入。
            payload.pop("thinking", None)
        thinking_payload = self.thinking_payload()
        if thinking_payload:
            payload.update(thinking_payload)
        provider_options = dict(self._defaults.get("provider_options") or {})
        provider_options.update(options.provider_options)
        if provider_options:
            payload.update(provider_options)
        # Anthropic 不接收 OpenAI 的 response_format；结构化输出由 LLM 提示词约束
        return payload

    def _response(self, data: Mapping[str, Any]) -> ChatModelResponse:
        """把 Messages 响应解析为统一 ChatModelResponse。"""
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
        """非流式调用 Messages API，并把 SDK 异常归一化为 ProviderError。"""
        payload = self._payload(request, stream=False)
        try:
            message = await self.provider.client.messages.create(
                **split_sdk_kwargs(payload, _ANTHROPIC_KWARGS)
            )
        except Exception as exc:
            raise sdk_error(
                exc,
                provider=self.provider.config.name,
                model=self.spec.model,
            ) from exc
        return self._response(sdk_dump(message))

    async def stream(self, request: ChatModelRequest) -> AsyncIterator[ChatStreamEvent]:
        """流式调用 Messages API，累积 tool_use JSON 并产出统一流事件。"""
        payload = self._payload(request, stream=True)
        yield ChatStreamStart(model=self.spec.model)
        blocks: dict[int, dict[str, Any]] = {}
        tool_json: dict[int, str] = {}
        usage: ChatUsage | None = None
        finish_reason = ChatFinishReason.STOP
        response_id: str | None = None
        try:
            stream = await self.provider.client.messages.create(
                **split_sdk_kwargs(payload, _ANTHROPIC_KWARGS)
            )
            async for event in stream:
                data = sdk_dump(event)
                if not isinstance(data, Mapping):
                    continue
                kind = data.get("type")
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
        except Exception as exc:
            yield ChatStreamError(
                error=sdk_error(
                    exc,
                    provider=self.provider.config.name,
                    model=self.spec.model,
                )
            )
            return
        yield ChatStreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
        )


class AnthropicChatProvider(ChatProvider):
    model_class = AnthropicChatModel
    default_capabilities = {
        "thinking": True,
        "vision": True,
        "file_input": True,
        "prefix_completion": True,
        "prompt_caching": True,
    }

    def __init__(self, config: ChatProviderConfig) -> None:
        """初始化 SDK 客户端占位与模型句柄缓存。"""
        super().__init__(config)
        self._client: anthropic.AsyncAnthropic | None = None
        self._handles: dict[str, ChatModel] = {}

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        """返回已初始化的 SDK 客户端；未 open 时抛 UNAVAILABLE。"""
        if self._client is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "Anthropic SDK 客户端尚未初始化",
                provider=self.config.name,
            )
        return self._client

    async def open(self) -> None:
        """构建并保存 AsyncAnthropic 客户端。"""
        client = build_anthropic_client(self.config)
        await super().open()
        self._client = client

    async def close(self) -> None:
        """关闭 SDK 客户端并清理模型句柄。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
        await super().close()

    def get_model(self, model_id: str) -> ChatModel:
        """按 model_id 复用或创建模型句柄。"""
        model = self._handles.get(model_id)
        if model is None:
            model = self.model_class(self.get_spec(model_id), self)
            self._handles[model_id] = model
        return model
