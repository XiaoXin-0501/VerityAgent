"""OpenAI Responses API adapter（item-based）。"""

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
    ResponseFormatType,
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

__all__ = ["OpenAIResponsesChatModel", "OpenAIResponsesChatProvider"]


def _usage(payload: Mapping[str, Any] | None) -> ChatUsage:
    data = payload or {}
    input_details = data.get("input_tokens_details") or {}
    output_details = data.get("output_tokens_details") or {}
    return ChatUsage(
        input_tokens=int(data.get("input_tokens") or 0),
        output_tokens=int(data.get("output_tokens") or 0),
        total_tokens=int(data.get("total_tokens") or 0),
        cached_input_tokens=int(input_details.get("cached_tokens") or 0),
        reasoning_tokens=int(output_details.get("reasoning_tokens") or 0),
    )


def _image_url(part: ModelImagePart) -> str:
    if part.url:
        return part.url
    if part.data:
        mime = part.mime_type or "image/png"
        return f"data:{mime};base64,{base64.b64encode(part.data).decode('ascii')}"
    raise ProviderError(ProviderErrorCode.INVALID_REQUEST, "图片 part 缺少 url/data")


def _content(parts: list[ModelContentPart], *, assistant: bool) -> list[dict[str, Any]]:
    text_type = "output_text" if assistant else "input_text"
    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ModelTextPart):
            result.append({"type": text_type, "text": part.text})
        elif isinstance(part, ModelImagePart):
            result.append({"type": "input_image", "image_url": _image_url(part)})
        elif isinstance(part, ModelFilePart):
            if part.text:
                result.append(
                    {
                        "type": text_type,
                        "text": f"[file {part.filename or ''}]\n{part.text}",
                    }
                )
            elif part.url:
                result.append({"type": "input_file", "file_url": part.url})
        elif isinstance(part, ModelThinkingPart) and not assistant and part.text:
            # Responses API 不接收历史 thinking；仅在需要时忽略
            continue
    return result


class OpenAIResponsesChatModel(ChatModel):
    def __init__(self, spec, provider: "OpenAIResponsesChatProvider") -> None:
        super().__init__(spec)
        self.provider = provider

    def _input(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.role is ModelRole.TOOL:
                text = "".join(
                    part.text
                    for part in message.content
                    if isinstance(part, ModelTextPart)
                )
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": text or json.dumps(_content(message.content, assistant=False)),
                    }
                )
                continue
            if message.role is ModelRole.ASSISTANT:
                text = "".join(
                    part.text
                    for part in message.content
                    if isinstance(part, ModelTextPart)
                )
                if text:
                    items.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text}],
                        }
                    )
                for call in message.tool_calls:
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": call.call_id,
                            "name": call.name,
                            "arguments": call.raw_arguments
                            or json.dumps(call.arguments or {}, ensure_ascii=False),
                        }
                    )
                continue
            items.append(
                {
                    "role": message.role.value,
                    "content": _content(message.content, assistant=False),
                }
            )
        return items

    def _tool_choice(self, request: ChatModelRequest) -> Any:
        choice = request.tool_choice
        if choice.mode is ToolChoiceMode.AUTO:
            return "auto"
        if choice.mode is ToolChoiceMode.NONE:
            return "none"
        if choice.mode is ToolChoiceMode.REQUIRED:
            return "required"
        return {"type": "function", "name": choice.tool_name}

    def _payload(self, request: ChatModelRequest, *, stream: bool) -> dict[str, Any]:
        if request.prefix:
            raise ProviderError(
                ProviderErrorCode.UNSUPPORTED_CAPABILITY,
                "OpenAI Responses provider 不支持 prefix completion",
                provider=self.provider.config.name,
                model=self.spec.model,
            )
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "input": self._input(request.messages),
            "stream": stream,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = self._tool_choice(request)
        options = request.options
        temperature = self.option("temperature", options.temperature)
        if temperature is not None:
            payload["temperature"] = temperature
        top_p = self.option("top_p", options.top_p)
        if top_p is not None:
            payload["top_p"] = top_p
        max_output_tokens = self.option(
            "max_output_tokens", options.max_output_tokens
        )
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        seed = self.option("seed", options.seed)
        if seed is not None:
            payload["seed"] = seed
        parallel_tool_calls = self.option(
            "parallel_tool_calls", options.parallel_tool_calls
        )
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        provider_options = dict(self._defaults.get("provider_options") or {})
        provider_options.update(options.provider_options)
        if provider_options:
            payload.update(provider_options)
        fmt = request.response_format
        if fmt.type is ResponseFormatType.JSON_OBJECT:
            payload["text"] = {"format": {"type": "json_object"}}
        elif fmt.type is ResponseFormatType.JSON_SCHEMA:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": fmt.name or "response",
                    "schema": fmt.schema_ or {},
                    **({"strict": fmt.strict} if fmt.strict is not None else {}),
                }
            }
        return payload

    def _finish(self, data: Mapping[str, Any], has_tools: bool) -> ChatFinishReason:
        if has_tools:
            return ChatFinishReason.TOOL_CALLS
        status = data.get("status")
        if status == "incomplete":
            reason = (data.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return ChatFinishReason.LENGTH
            if reason == "content_filter":
                return ChatFinishReason.CONTENT_FILTER
        if status == "failed":
            return ChatFinishReason.ERROR
        return ChatFinishReason.STOP

    def _response(self, data: Mapping[str, Any]) -> ChatModelResponse:
        parts: list[ModelContentPart] = []
        calls: list[ModelToolCall] = []
        for item in data.get("output") or []:
            if not isinstance(item, Mapping):
                continue
            kind = item.get("type")
            if kind == "message":
                for block in item.get("content") or []:
                    if not isinstance(block, Mapping):
                        continue
                    if block.get("type") == "output_text":
                        parts.append(ModelTextPart(text=str(block.get("text") or "")))
                    elif block.get("type") == "refusal":
                        parts.append(ModelTextPart(text=str(block.get("refusal") or "")))
            elif kind == "reasoning":
                for summary in item.get("summary") or []:
                    if isinstance(summary, Mapping) and summary.get("text"):
                        parts.append(ModelThinkingPart(text=str(summary["text"])))
            elif kind == "function_call":
                raw = item.get("arguments")
                arguments = None
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                        arguments = parsed if isinstance(parsed, dict) else None
                    except ValueError:
                        arguments = None
                calls.append(
                    ModelToolCall(
                        call_id=str(item.get("call_id") or item.get("id") or ""),
                        name=str(item.get("name") or ""),
                        arguments=arguments,
                        raw_arguments=raw if isinstance(raw, str) else None,
                    )
                )
        return ChatModelResponse(
            message=ModelMessage(
                role=ModelRole.ASSISTANT, content=parts, tool_calls=calls
            ),
            finish_reason=self._finish(data, bool(calls)),
            usage=_usage(data.get("usage")),
            response_id=data.get("id"),
        )

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        data = await self.provider.request_json(
            "POST",
            "responses",
            payload=self._payload(request, stream=False),
            model=self.spec.model,
        )
        return self._response(data)

    async def stream(self, request: ChatModelRequest) -> AsyncIterator[ChatStreamEvent]:
        payload = self._payload(request, stream=True)
        yield ChatStreamStart(model=self.spec.model)
        calls: dict[str, ModelToolCall] = {}
        usage: ChatUsage | None = None
        finish_reason = ChatFinishReason.STOP
        response_id: str | None = None
        try:
            async for event_name, data_text in self.provider.stream_sse(
                "responses", payload=payload, model=self.spec.model
            ):
                try:
                    data = json.loads(data_text)
                except ValueError:
                    continue
                kind = data.get("type") or event_name
                if kind in ("error", "response.failed"):
                    error = data.get("error") or data.get("response", {}).get("error") or {}
                    raise ProviderError(
                        ProviderErrorCode.UPSTREAM,
                        str(error.get("message") or "OpenAI Responses stream error"),
                        provider=self.provider.config.name,
                        model=self.spec.model,
                    )
                if kind == "response.created":
                    response_id = (data.get("response") or {}).get("id")
                elif kind == "response.output_text.delta":
                    yield ChatTextDelta(text=str(data.get("delta") or ""))
                elif kind == "response.reasoning_summary_text.delta":
                    yield ChatThinkingDelta(text=str(data.get("delta") or ""))
                elif kind == "response.output_item.added":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        call_id = str(item.get("call_id") or item.get("id") or "")
                        calls[call_id] = ModelToolCall(
                            call_id=call_id,
                            name=str(item.get("name") or ""),
                        )
                        yield ChatToolCallStart(
                            call_id=call_id,
                            name=str(item.get("name") or ""),
                            index=int(data.get("output_index") or 0),
                        )
                elif kind == "response.function_call_arguments.delta":
                    call_id = str(data.get("item_id") or data.get("call_id") or "")
                    call = calls.get(call_id)
                    fragment = str(data.get("delta") or "")
                    if call is not None:
                        call.raw_arguments = (call.raw_arguments or "") + fragment
                        yield ChatToolCallDelta(
                            call_id=call_id,
                            arguments_delta=fragment,
                            index=int(data.get("output_index") or 0),
                        )
                elif kind == "response.output_item.done":
                    item = data.get("item") or {}
                    if item.get("type") == "function_call":
                        call_id = str(item.get("call_id") or item.get("id") or "")
                        call = calls.get(call_id)
                        raw = item.get("arguments")
                        arguments = None
                        if isinstance(raw, str):
                            try:
                                parsed = json.loads(raw)
                                arguments = parsed if isinstance(parsed, dict) else None
                            except ValueError:
                                arguments = None
                        yield ChatToolCallEnd(
                            call_id=call_id,
                            arguments=arguments,
                            raw_arguments=raw if isinstance(raw, str) else None,
                            index=int(data.get("output_index") or 0),
                        )
                elif kind == "response.completed":
                    response = data.get("response") or {}
                    response_id = response.get("id") or response_id
                    usage = _usage(response.get("usage"))
                    if usage.total_tokens:
                        yield ChatUsageEvent(usage=usage)
                    finish_reason = self._finish(response, bool(calls))
                    break
        except ProviderError as exc:
            yield ChatStreamError(error=exc)
            return
        yield ChatStreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
        )


class OpenAIResponsesChatProvider(ChatProvider, HttpProviderMixin):
    model_class = OpenAIResponsesChatModel
    default_capabilities = {
        "tool_calling": True,
        "parallel_tool_calls": True,
        "vision": True,
        "file_input": True,
        "prompt_caching": True,
    }

    def __init__(self, config: ChatProviderConfig) -> None:
        super().__init__(config)
        self._init_http()
        self._handles: dict[str, ChatModel] = {}

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
