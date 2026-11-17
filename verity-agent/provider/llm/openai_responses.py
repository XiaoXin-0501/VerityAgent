"""OpenAI Responses API adapter（item-based）。"""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import openai

from config.provider import ChatProviderConfig
from provider.enums import (
    ChatFinishReason,
    ModelRole,
    ResponseFormatType,
    ToolChoiceMode,
)
from exception.provider import ProviderError, ProviderErrorCode
from provider.llm.base import ChatModel, ChatProvider
from provider.sdk import (
    build_openai_client,
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

__all__ = ["OpenAIResponsesChatModel", "OpenAIResponsesChatProvider"]

_RESPONSES_KWARGS = frozenset(
    {
        "model",
        "input",
        "stream",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_output_tokens",
        "parallel_tool_calls",
        "text",
    }
)


def _usage(payload: Mapping[str, Any] | None) -> ChatUsage:
    """从 Responses usage 构造统一 token 用量。"""
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
    """把图片 part 转成 URL 或 data URI。"""
    if part.url:
        return part.url
    if part.data:
        mime = part.mime_type or "image/png"
        return f"data:{mime};base64,{base64.b64encode(part.data).decode('ascii')}"
    raise ProviderError(ProviderErrorCode.INVALID_REQUEST, "图片 part 缺少 url/data")


def _content(parts: list[ModelContentPart], *, assistant: bool) -> list[dict[str, Any]]:
    """把统一 content part 转为 Responses content；assistant 用 output_text。"""
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
        """绑定模型规格与所属 Provider。"""
        super().__init__(spec)
        self.provider = provider

    def _input(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """把统一消息转换为 Responses item 列表（function_call/output 独立成项）。"""
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
        """把统一工具选择策略转换为 Responses tool_choice。"""
        choice = request.tool_choice
        if choice.mode is ToolChoiceMode.AUTO:
            return "auto"
        if choice.mode is ToolChoiceMode.NONE:
            return "none"
        if choice.mode is ToolChoiceMode.REQUIRED:
            return "required"
        return {"type": "function", "name": choice.tool_name}

    def _payload(self, request: ChatModelRequest, *, stream: bool) -> dict[str, Any]:
        """构造 Responses 请求体；不支持 prefix completion。"""
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
        thinking_payload = self.thinking_payload()
        if thinking_payload:
            payload.update(thinking_payload)
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
        """根据 status/tool 输出推导统一结束原因。"""
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
        """把 Responses 输出项解析为统一 ChatModelResponse。"""
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
        """非流式调用 Responses API，并归一化 SDK 异常。"""
        payload = self._payload(request, stream=False)
        try:
            response = await self.provider.client.responses.create(
                **split_sdk_kwargs(payload, _RESPONSES_KWARGS)
            )
        except Exception as exc:
            raise sdk_error(
                exc,
                provider=self.provider.config.name,
                model=self.spec.model,
            ) from exc
        return self._response(sdk_dump(response))

    async def stream(self, request: ChatModelRequest) -> AsyncIterator[ChatStreamEvent]:
        """流式消费 Responses 事件，累积函数调用参数并产出统一事件。"""
        payload = self._payload(request, stream=True)
        yield ChatStreamStart(model=self.spec.model)
        calls: dict[str, ModelToolCall] = {}
        usage: ChatUsage | None = None
        finish_reason = ChatFinishReason.STOP
        response_id: str | None = None
        try:
            stream = await self.provider.client.responses.create(
                **split_sdk_kwargs(payload, _RESPONSES_KWARGS)
            )
            async for event in stream:
                data = sdk_dump(event)
                if not isinstance(data, Mapping):
                    continue
                kind = data.get("type")
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


class OpenAIResponsesChatProvider(ChatProvider):
    model_class = OpenAIResponsesChatModel
    default_capabilities = {
        "tool_calling": True,
        "parallel_tool_calls": True,
        "vision": True,
        "file_input": True,
        "prompt_caching": True,
    }

    def __init__(self, config: ChatProviderConfig) -> None:
        """初始化 SDK 客户端占位与模型句柄缓存。"""
        super().__init__(config)
        self._client: openai.AsyncOpenAI | None = None
        self._handles: dict[str, ChatModel] = {}

    @property
    def client(self) -> openai.AsyncOpenAI:
        """返回已初始化的 SDK 客户端；未 open 时抛 UNAVAILABLE。"""
        if self._client is None:
            raise ProviderError(
                ProviderErrorCode.UNAVAILABLE,
                "OpenAI SDK 客户端尚未初始化",
                provider=self.config.name,
            )
        return self._client

    async def open(self) -> None:
        """构建并保存 AsyncOpenAI 客户端。"""
        client = build_openai_client(self.config)
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
