"""Provider ChatModel -> Agent ``LLMPort`` 的最小适配器。

Agent framework 以 ``agent.llm.LLMRequest/LLMResponse`` 为边界；
本适配器只做 wire 转换，不负责 PromptBuilder、模型 fallback、ContextPlan。
这些能力由 ``agent.prompting`` / ``agent.llm_service`` / ``agent.context`` 负责，
不要在这里复制一套。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from agent.llm import (
    FilePart,
    FinishReason,
    ImagePart,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMStreamEvent,
    LLMToolCall,
    LLMUsage,
    TextPart,
    ThinkingPart,
)
from exception import ConfigurationError
from exception.provider import ProviderError, ProviderErrorCode
from provider.enums import ChatFinishReason, ModelRole
from provider.llm.types import (
    ChatModelRequest,
    ChatStreamEnd,
    ChatStreamError,
    ChatTextDelta,
    ChatThinkingDelta,
    ChatToolCallDelta,
    ChatToolCallEnd,
    ChatToolCallStart,
    ChatUsageEvent,
    ModelFilePart,
    ModelGenerationOptions,
    ModelImagePart,
    ModelMessage,
    ModelResponseFormat,
    ModelTextPart,
    ModelThinkingPart,
    ModelTool,
    ModelToolCall,
    ModelToolChoice,
)

__all__ = ["ProviderChatAdapter", "RoutingChatAdapter"]


class ProviderChatAdapter:
    """把一个已初始化的 ChatModel 句柄适配成 Agent LLMPort。"""

    def __init__(self, model: Any, *, model_id: str | None = None) -> None:
        """保存模型句柄；模型生命周期由 ProviderRegistry 管理。"""
        if not hasattr(model, "complete"):
            raise TypeError("ChatModel 必须提供 complete()")
        self.model = model
        self.model_id = model_id or getattr(getattr(model, "spec", None), "id", "")
        if not self.model_id:
            raise ValueError("ProviderChatAdapter 缺少 model_id")

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """调用 provider 非流式接口，并映射回 Agent LLMResponse。"""
        if request.model_id and request.model_id != self.model_id:
            raise ConfigurationError(
                f"ProviderChatAdapter 绑定模型 {self.model_id}，"
                f"但请求要求 {request.model_id}",
                details={"model_id": request.model_id},
            )
        response = await self.model.complete(self._request(request))
        return self._response(request, response)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """调用 provider 流式接口，并归一为 Agent 应用层事件。"""
        if request.model_id and request.model_id != self.model_id:
            raise ConfigurationError(
                f"ProviderChatAdapter 绑定模型 {self.model_id}，"
                f"但请求要求 {request.model_id}",
                details={"model_id": request.model_id},
            )
        provider_request = self._request(request).model_copy(update={"stream": True})
        try:
            async for event in self.model.stream(provider_request):
                if isinstance(event, ChatTextDelta):
                    yield LLMStreamEvent(
                        type="text_delta",
                        text=event.text,
                        index=event.index,
                    )
                elif isinstance(event, ChatThinkingDelta):
                    yield LLMStreamEvent(
                        type="thinking_delta",
                        text=event.text,
                        index=event.index,
                    )
                elif isinstance(event, ChatToolCallStart):
                    yield LLMStreamEvent(
                        type="tool_call_start",
                        call_id=event.call_id,
                        name=event.name,
                        index=event.index,
                    )
                elif isinstance(event, ChatToolCallDelta):
                    yield LLMStreamEvent(
                        type="tool_call_delta",
                        call_id=event.call_id,
                        arguments_delta=event.arguments_delta,
                        index=event.index,
                    )
                elif isinstance(event, ChatToolCallEnd):
                    yield LLMStreamEvent(
                        type="tool_call_end",
                        call_id=event.call_id,
                        arguments=dict(event.arguments or {}),
                        raw_arguments=event.raw_arguments,
                        index=event.index,
                    )
                elif isinstance(event, ChatUsageEvent):
                    yield LLMStreamEvent(
                        type="usage",
                        usage=_usage(event.usage),
                    )
                elif isinstance(event, ChatStreamEnd):
                    yield LLMStreamEvent(
                        type="end",
                        finish_reason=_finish_reason(event.finish_reason),
                        usage=_usage(event.usage) if event.usage else None,
                    )
                elif isinstance(event, ChatStreamError):
                    yield LLMStreamEvent(
                        type="error",
                        error_code=str(getattr(event.error, "code", "PROVIDER_ERROR")),
                        error_message=str(event.error),
                    )
                    return
        except Exception as exc:  # noqa: BLE001 - 流式端口异常收敛
            yield LLMStreamEvent(
                type="error",
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

    def _request(self, request: LLMRequest) -> ChatModelRequest:
        """LLMRequest -> provider ChatModelRequest。"""
        return ChatModelRequest(
            model=self.model_id,
            messages=[_provider_message(message) for message in request.messages],
            tools=[
                ModelTool(
                    name=item.name,
                    description=item.description,
                    parameters=item.parameters,
                    strict=item.strict,
                )
                for item in request.tools
            ],
            tool_choice=ModelToolChoice(
                mode=request.tool_choice.mode,
                tool_name=request.tool_choice.tool_name,
            ),
            options=ModelGenerationOptions(
                temperature=request.options.temperature,
                top_p=request.options.top_p,
                max_output_tokens=request.options.max_output_tokens,
                stop=list(request.options.stop),
                seed=request.options.seed,
                presence_penalty=request.options.presence_penalty,
                frequency_penalty=request.options.frequency_penalty,
                thinking_budget=request.options.thinking_budget,
                parallel_tool_calls=request.options.parallel_tool_calls,
                provider_options=dict(request.options.provider_options),
            ),
            response_format=ModelResponseFormat(
                type=request.response_format.type,
                name=request.response_format.name,
                schema=request.response_format.schema_,
                strict=request.response_format.strict,
            ),
            stream=False,
            prefix=request.prefix_completion,
        )

    def _response(self, request: LLMRequest, response: Any) -> LLMResponse:
        """provider ChatModelResponse -> Agent LLMResponse。"""
        message = response.message
        parts: list[Any] = []
        for part in message.content:
            if isinstance(part, ModelTextPart):
                parts.append(TextPart(text=part.text))
            elif isinstance(part, ModelThinkingPart):
                parts.append(
                    ThinkingPart(
                        text=part.text,
                        signature=part.signature,
                        redacted=part.redacted,
                    )
                )
            # provider 的图片/文件回传不是当前 Agent 输出契约的一部分，忽略。
        tool_calls = [
            LLMToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=dict(call.arguments or {}),
                raw_arguments=call.raw_arguments,
                index=call.index,
            )
            for call in message.tool_calls
        ]
        return LLMResponse(
            request_id=request.request_id,
            provider=str(getattr(response, "provider", "") or self.model.spec.provider),
            model=str(getattr(response, "model", "") or self.model.spec.model),
            message=LLMMessage(
                role=LLMRole.ASSISTANT,
                content=parts,
                tool_calls=tool_calls,
            ),
            finish_reason=_finish_reason(response.finish_reason),
            usage=_usage(response.usage),
            response_id=getattr(response, "response_id", None),
        )


class RoutingChatAdapter:
    """按 ``LLMRequest.model_id`` 路由到 Provider 模型句柄的 LLMPort。

    Agent framework 只有一个默认模型时，它等价于 ``ProviderChatAdapter``；
    请求显式覆盖 ``model_id`` 时，从 ProviderRegistry 解析对应 chat 模型。
    模型白名单与 fallback 由外层 ``agent.llm_service.ChatService/ModelRegistry``
    负责，这里只做路由和类型校验。
    """

    def __init__(
        self,
        provider_registry: Any,
        *,
        default_model_id: str,
    ) -> None:
        """保存注册表与默认模型；句柄按需解析并缓存。"""
        if not default_model_id:
            raise ValueError("RoutingChatAdapter 缺少 default_model_id")
        self.provider_registry = provider_registry
        self.default_model_id = default_model_id
        self._adapters: dict[str, ProviderChatAdapter] = {}

    async def chat(self, request: LLMRequest) -> LLMResponse:
        """按请求模型 id 选择 adapter；未知/禁用/非 chat 模型明确报错。"""
        model_id = request.model_id or self.default_model_id
        adapter = self._adapters.get(model_id)
        if adapter is None:
            adapter = self._resolve(model_id)
            self._adapters[model_id] = adapter
        return await adapter.chat(request)

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        """按请求模型 id 路由流式调用。"""
        model_id = request.model_id or self.default_model_id
        adapter = self._adapters.get(model_id)
        if adapter is None:
            adapter = self._resolve(model_id)
            self._adapters[model_id] = adapter
        async for event in adapter.stream(request):
            yield event

    def _resolve(self, model_id: str) -> ProviderChatAdapter:
        """从 ProviderRegistry 解析 chat 模型句柄。"""
        try:
            model = self.provider_registry.get_model_by_id(model_id)
        except ConfigurationError:
            raise
        spec = getattr(model, "spec", None)
        kind = getattr(spec, "kind", None)
        kind_value = getattr(kind, "value", kind)
        if kind_value is not None and kind_value != "chat":
            raise ConfigurationError(
                f"请求模型不是 chat 模型: {model_id} (kind={kind_value})",
                details={"model_id": model_id},
            )
        return ProviderChatAdapter(model, model_id=model_id)


def _provider_message(message: LLMMessage) -> ModelMessage:
    """Agent LLMMessage -> provider ModelMessage。"""
    content = []
    for part in message.content:
        if isinstance(part, TextPart):
            content.append(ModelTextPart(text=part.text))
        elif isinstance(part, ThinkingPart):
            content.append(
                ModelThinkingPart(
                    text=part.text,
                    signature=part.signature,
                    redacted=part.redacted,
                )
            )
        elif isinstance(part, ImagePart) and (part.url or part.data):
            content.append(
                ModelImagePart(
                    url=part.url,
                    data=part.data,
                    mime_type=part.mime_type,
                    detail=part.detail,
                )
            )
        elif isinstance(part, FilePart):
            if not (part.url or part.text):
                # 上层必须先把 file_id 解析成文本/URL；静默跳过会让模型收到
                # 空上下文，问题更难排查。
                raise ProviderError(
                    ProviderErrorCode.INVALID_REQUEST,
                    "FilePart 的 file_id 尚未解析为 url/text",
                    details={"file_id": str(part.file_id) if part.file_id else None},
                )
            content.append(
                ModelFilePart(
                    url=part.url,
                    text=part.text,
                    filename=part.filename,
                    mime_type=part.mime_type,
                )
            )
    return ModelMessage(
        role=ModelRole(message.role.value),
        content=content,
        name=message.name,
        tool_calls=[
            ModelToolCall(
                call_id=call.call_id,
                name=call.name,
                arguments=call.arguments,
                raw_arguments=call.raw_arguments,
                index=call.index,
            )
            for call in message.tool_calls
        ],
        tool_call_id=message.tool_call_id,
        prefix=message.prefix,
        cacheable=message.cacheable,
    )


def _finish_reason(reason: ChatFinishReason) -> FinishReason:
    """Provider finish reason -> Agent finish reason。"""
    return FinishReason(reason.value)


def _usage(value: Any) -> LLMUsage:
    """Provider usage -> Agent LLMUsage；缺失时返回默认值。"""
    if value is None:
        return LLMUsage()
    return LLMUsage(
        input_tokens=int(getattr(value, "input_tokens", 0) or 0),
        output_tokens=int(getattr(value, "output_tokens", 0) or 0),
        total_tokens=int(getattr(value, "total_tokens", 0) or 0),
        cached_input_tokens=int(getattr(value, "cached_input_tokens", 0) or 0),
        reasoning_tokens=int(getattr(value, "reasoning_tokens", 0) or 0),
        source=getattr(value, "source", None) or LLMUsage().source,
    )
