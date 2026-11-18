"""OpenAI Chat Completions 兼容 adapter。

覆盖 OpenAI / DeepSeek / vLLM / llama.cpp 等 OpenAI-compatible 接口；
厂商差异通过子类覆盖 ``_apply_prefix`` 等 hook 实现。
"""

from __future__ import annotations

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
from provider.llm.helpers import image_part_url
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

__all__ = [
    "OpenAIChatModel",
    "OpenAIChatProvider",
    "DeepSeekChatModel",
    "DeepSeekChatProvider",
    "VLLMChatModel",
    "VLLMChatProvider",
]

_CHAT_KWARGS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "tools",
        "tool_choice",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "parallel_tool_calls",
        "response_format",
        "stream_options",
    }
)

_FINISH_MAP = {
    "stop": ChatFinishReason.STOP,
    "length": ChatFinishReason.LENGTH,
    "tool_calls": ChatFinishReason.TOOL_CALLS,
    "function_call": ChatFinishReason.TOOL_CALLS,
    "content_filter": ChatFinishReason.CONTENT_FILTER,
    "error": ChatFinishReason.ERROR,
    "cancelled": ChatFinishReason.CANCELLED,
}


def _finish_reason(value: Any) -> ChatFinishReason:
    """把 OpenAI finish_reason 归一化为统一结束原因。"""
    return _FINISH_MAP.get(str(value), ChatFinishReason.STOP)


def _parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    """解析 tool_call.arguments：返回解析后的 dict 与原始字符串，便于回放。"""
    if isinstance(raw, dict):
        return raw, None
    if raw is None:
        return None, None
    text = str(raw)
    try:
        value = json.loads(text)
    except ValueError:
        return None, text
    return (value if isinstance(value, dict) else None), text


def _usage(data: Mapping[str, Any] | None) -> ChatUsage:
    """从 usage 字典构造统一 token 用量（含缓存与推理 token）。"""
    payload = data or {}
    prompt_details = payload.get("prompt_tokens_details") or {}
    completion_details = payload.get("completion_tokens_details") or {}
    return ChatUsage(
        input_tokens=int(payload.get("prompt_tokens") or payload.get("input_tokens") or 0),
        output_tokens=int(
            payload.get("completion_tokens") or payload.get("output_tokens") or 0
        ),
        total_tokens=int(payload.get("total_tokens") or 0),
        cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
        reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
    )


def _content_parts(parts: list[ModelContentPart]) -> str | list[dict[str, Any]]:
    """纯文本压成 string，多模态才用 parts 数组。"""
    texts = [part.text for part in parts if isinstance(part, ModelTextPart)]
    if len(texts) == len(parts):
        return "".join(texts)

    result: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, ModelTextPart):
            result.append({"type": "text", "text": part.text})
        elif isinstance(part, ModelImagePart):
            result.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_part_url(part),
                        **({"detail": part.detail} if part.detail else {}),
                    },
                }
            )
        elif isinstance(part, ModelFilePart):
            if part.text:
                result.append(
                    {
                        "type": "text",
                        "text": f"[file {part.filename or ''}]\n{part.text}",
                    }
                )
            elif part.url:
                result.append({"type": "text", "text": part.url})
        elif isinstance(part, ModelThinkingPart):
            # OpenAI chat 不接收历史 thinking 块
            continue
    return result


def _tool_choice(choice) -> Any:
    """把统一工具选择策略转换为 Chat Completions tool_choice。"""
    if choice.mode is ToolChoiceMode.AUTO:
        return "auto"
    if choice.mode is ToolChoiceMode.NONE:
        return "none"
    if choice.mode is ToolChoiceMode.REQUIRED:
        return "required"
    return {"type": "function", "function": {"name": choice.tool_name}}


class OpenAIChatModel(ChatModel):
    def __init__(self, spec, provider: "OpenAIChatProvider") -> None:
        """绑定模型规格与所属 Provider。"""
        super().__init__(spec)
        self.provider = provider

    # ——————————————————— request 映射 ———————————————————

    def _messages(self, messages: list[ModelMessage]) -> list[dict[str, Any]]:
        """把统一消息转换为 Chat Completions messages，保留 tool_calls/name。"""
        result: list[dict[str, Any]] = []
        for message in messages:
            if message.role is ModelRole.ASSISTANT:
                item: dict[str, Any] = {
                    "role": "assistant",
                    "content": _content_parts(message.content),
                }
                if message.tool_calls:
                    item["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.raw_arguments
                                or json.dumps(call.arguments or {}, ensure_ascii=False),
                            },
                        }
                        for call in message.tool_calls
                    ]
                result.append(item)
            elif message.role is ModelRole.TOOL:
                text = _content_parts(message.content)
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": text if isinstance(text, str) else json.dumps(text),
                    }
                )
            else:
                item = {
                    "role": message.role.value,
                    "content": _content_parts(message.content),
                }
                if message.name:
                    item["name"] = message.name
                result.append(item)
        return result

    def _response_format(self, request: ChatModelRequest) -> dict[str, Any] | None:
        """按响应格式配置生成 response_format；TEXT 不传该字段。"""
        fmt = request.response_format
        if fmt.type is ResponseFormatType.TEXT:
            return None
        if fmt.type is ResponseFormatType.JSON_OBJECT:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.name or "response",
                "schema": fmt.schema_ or {},
                **({"strict": fmt.strict} if fmt.strict is not None else {}),
            },
        }

    def _apply_prefix(self, payload: dict[str, Any], request: ChatModelRequest) -> None:
        # OpenAI Chat 没有原生 prefix；由子类实现 DeepSeek/vLLM 的机制
        """默认不支持 prefix completion；厂商子类按各自机制覆盖。"""
        raise ProviderError(
            ProviderErrorCode.UNSUPPORTED_CAPABILITY,
            "当前 OpenAI Chat provider 不支持 prefix completion",
            provider=self.provider.config.name,
            model=self.spec.model,
        )

    def _payload(self, request: ChatModelRequest, *, stream: bool) -> dict[str, Any]:
        """构造 Chat Completions 请求体，统一映射采样参数与工具定义。"""
        payload: dict[str, Any] = {
            "model": self.spec.model,
            "messages": self._messages(request.messages),
            "stream": stream,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = _tool_choice(request.tool_choice)
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
            key = (
                "max_completion_tokens"
                if self.provider.config.metadata.get("use_max_completion_tokens")
                else "max_tokens"
            )
            payload[key] = max_output_tokens
        stop = options.stop or self._defaults.get("stop") or []
        if stop:
            payload["stop"] = stop
        seed = self.option("seed", options.seed)
        if seed is not None:
            payload["seed"] = seed
        presence_penalty = self.option(
            "presence_penalty", options.presence_penalty
        )
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        frequency_penalty = self.option(
            "frequency_penalty", options.frequency_penalty
        )
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        parallel_tool_calls = self.option(
            "parallel_tool_calls", options.parallel_tool_calls
        )
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = parallel_tool_calls
        provider_options = dict(self._defaults.get("provider_options") or {})
        provider_options.update(options.provider_options)
        if provider_options:
            payload.update(provider_options)
        # 模型级 thinking 开关：关闭时发送厂商的关闭参数，而不是只丢弃
        # reasoning 内容；未配置关闭参数时保持厂商默认。
        thinking_payload = self.thinking_payload()
        if thinking_payload:
            payload.update(thinking_payload)
        fmt = self._response_format(request)
        if fmt is not None:
            payload["response_format"] = fmt
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if request.prefix:
            self._apply_prefix(payload, request)
        return payload

    # ——————————————————— response 映射 ———————————————————

    def _response_message(self, message: Mapping[str, Any]) -> ModelMessage:
        """把 Chat Completions message 解析为统一 ModelMessage。"""
        parts: list[ModelContentPart] = []
        content = message.get("content")
        if isinstance(content, str) and content:
            parts.append(ModelTextPart(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") in (
                    "text",
                    "output_text",
                ):
                    text = item.get("text")
                    if text:
                        parts.append(ModelTextPart(text=str(text)))
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning:
            parts.append(ModelThinkingPart(text=str(reasoning)))

        calls: list[ModelToolCall] = []
        for item in message.get("tool_calls") or []:
            if not isinstance(item, Mapping):
                continue
            function = item.get("function") or {}
            arguments, raw = _parse_arguments(function.get("arguments"))
            calls.append(
                ModelToolCall(
                    call_id=str(item.get("id") or ""),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                    raw_arguments=raw,
                )
            )
        return ModelMessage(
            role=ModelRole.ASSISTANT,
            content=parts,
            tool_calls=calls,
        )

    async def complete(self, request: ChatModelRequest) -> ChatModelResponse:
        """非流式调用 Chat Completions，并校验 choices 非空。"""
        payload = self._payload(request, stream=False)
        try:
            completion = await self.provider.client.chat.completions.create(
                **split_sdk_kwargs(payload, _CHAT_KWARGS)
            )
        except Exception as exc:
            raise sdk_error(
                exc,
                provider=self.provider.config.name,
                model=self.spec.model,
            ) from exc
        data = sdk_dump(completion)
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                ProviderErrorCode.RESPONSE_INVALID,
                "Provider 响应缺少 choices",
                provider=self.provider.config.name,
                model=self.spec.model,
            )
        choice = choices[0] or {}
        return ChatModelResponse(
            message=self._response_message(choice.get("message") or {}),
            finish_reason=_finish_reason(choice.get("finish_reason")),
            usage=_usage(data.get("usage")),
            response_id=data.get("id"),
            metadata={"model": data.get("model")},
        )

    async def stream(self, request: ChatModelRequest) -> AsyncIterator[ChatStreamEvent]:
        """流式调用 Chat Completions，累积工具参数并在结束时统一产出。"""
        payload = self._payload(request, stream=True)
        yield ChatStreamStart(model=self.spec.model)

        tool_calls: dict[int, ModelToolCall] = {}
        started: set[int] = set()
        usage: ChatUsage | None = None
        finish_reason = ChatFinishReason.STOP
        response_id: str | None = None

        try:
            stream = await self.provider.client.chat.completions.create(
                **split_sdk_kwargs(payload, _CHAT_KWARGS)
            )
            async for chunk in stream:
                data = sdk_dump(chunk)
                if not isinstance(data, Mapping):
                    continue
                if data.get("error"):
                    raise ProviderError(
                        ProviderErrorCode.UPSTREAM,
                        str(data["error"]),
                        provider=self.provider.config.name,
                        model=self.spec.model,
                    )
                response_id = response_id or data.get("id")
                if data.get("usage"):
                    usage = _usage(data.get("usage"))
                    yield ChatUsageEvent(usage=usage)
                for choice in data.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield ChatThinkingDelta(text=str(reasoning))
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        yield ChatTextDelta(text=content)
                    for item in delta.get("tool_calls") or []:
                        if not isinstance(item, Mapping):
                            continue
                        index = int(item.get("index") or 0)
                        function = item.get("function") or {}
                        call_id = str(item.get("id") or f"call_{index}")
                        if index not in started:
                            started.add(index)
                            tool_calls[index] = ModelToolCall(
                                call_id=call_id,
                                name=str(function.get("name") or ""),
                                index=index,
                            )
                            yield ChatToolCallStart(
                                call_id=call_id,
                                name=str(function.get("name") or ""),
                                index=index,
                            )
                        arguments_delta = function.get("arguments")
                        if arguments_delta:
                            call = tool_calls[index]
                            call.raw_arguments = (call.raw_arguments or "") + str(
                                arguments_delta
                            )
                            yield ChatToolCallDelta(
                                call_id=call_id,
                                arguments_delta=str(arguments_delta),
                                index=index,
                            )
                    if choice.get("finish_reason"):
                        finish_reason = _finish_reason(choice.get("finish_reason"))
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

        for index, call in sorted(tool_calls.items()):
            arguments, _raw = _parse_arguments(call.raw_arguments)
            yield ChatToolCallEnd(
                call_id=call.call_id,
                arguments=arguments,
                raw_arguments=call.raw_arguments,
                index=index,
            )
        yield ChatStreamEnd(
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
        )


class OpenAIChatProvider(ChatProvider):
    """OpenAI Chat Completions 协议 provider（官方 SDK）。"""

    model_class = OpenAIChatModel
    default_capabilities: dict[str, Any] = {}

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


class DeepSeekChatModel(OpenAIChatModel):
    def _apply_prefix(self, payload: dict[str, Any], request: ChatModelRequest) -> None:
        """DeepSeek 以 prefix=true 开启续写模式。"""
        payload["prefix"] = True


class DeepSeekChatProvider(OpenAIChatProvider):
    model_class = DeepSeekChatModel
    default_capabilities = {"prefix_completion": True}


class VLLMChatModel(OpenAIChatModel):
    def _apply_prefix(self, payload: dict[str, Any], request: ChatModelRequest) -> None:
        """vLLM 通过 continue_final_message 续写最后一条 assistant 消息。"""
        payload["continue_final_message"] = True
        payload["add_generation_prompt"] = False


class VLLMChatProvider(OpenAIChatProvider):
    model_class = VLLMChatModel
    default_capabilities = {"prefix_completion": True}
