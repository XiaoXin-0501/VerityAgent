"""Chat provider 的 wire 模型（厂商无关）。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from provider.enums import (
    ChatFinishReason,
    ModelRole,
    ResponseFormatType,
    ToolChoiceMode,
    UsageSource,
)
from exception.provider import ProviderError
from utils.time import utcnow

__all__ = [
    "ModelTextPart",
    "ModelImagePart",
    "ModelFilePart",
    "ModelThinkingPart",
    "ModelContentPart",
    "ModelToolCall",
    "ModelMessage",
    "ModelTool",
    "ModelToolChoice",
    "ModelGenerationOptions",
    "ModelResponseFormat",
    "ChatUsage",
    "ChatModelRequest",
    "ChatModelResponse",
    "ChatStreamStart",
    "ChatTextDelta",
    "ChatThinkingDelta",
    "ChatToolCallStart",
    "ChatToolCallDelta",
    "ChatToolCallEnd",
    "ChatUsageEvent",
    "ChatStreamEnd",
    "ChatStreamError",
    "ChatStreamEvent",
]


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelTextPart(WireModel):
    type: Literal["text"] = "text"
    text: str


class ModelImagePart(WireModel):
    type: Literal["image"] = "image"
    url: str | None = None
    data: bytes | None = None
    mime_type: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _has_source(self) -> "ModelImagePart":
        if not self.url and not self.data:
            raise ValueError("图片 part 必须提供 url 或 data")
        return self


class ModelFilePart(WireModel):
    type: Literal["file"] = "file"
    url: str | None = None
    text: str | None = None
    filename: str | None = None
    mime_type: str | None = None

    @model_validator(mode="after")
    def _has_source(self) -> "ModelFilePart":
        if not self.url and not self.text:
            raise ValueError("文件 part 必须提供 url 或 text")
        return self


class ModelThinkingPart(WireModel):
    type: Literal["thinking"] = "thinking"
    text: str = ""
    signature: str | None = None
    redacted: bool = False


ModelContentPart = ModelTextPart | ModelImagePart | ModelFilePart | ModelThinkingPart


class ModelToolCall(WireModel):
    call_id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None
    index: int | None = None


class ModelMessage(WireModel):
    role: ModelRole
    content: list[ModelContentPart] = Field(default_factory=list)
    name: str | None = None
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    # 派生标志：仅当前请求最后一条 assistant 且要续写（D-41）
    prefix: bool = False
    cacheable: bool = False


class ModelTool(WireModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False


class ModelToolChoice(WireModel):
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    tool_name: str | None = None

    @model_validator(mode="after")
    def _tool_name_required(self) -> "ModelToolChoice":
        if self.mode is ToolChoiceMode.TOOL and not self.tool_name:
            raise ValueError("tool_choice=TOOL 时必须提供 tool_name")
        return self


class ModelGenerationOptions(WireModel):
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    thinking_budget: int | None = None
    parallel_tool_calls: bool | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class ModelResponseFormat(WireModel):
    type: ResponseFormatType = ResponseFormatType.TEXT
    name: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None

    @model_validator(mode="after")
    def _schema_required(self) -> "ModelResponseFormat":
        if self.type is ResponseFormatType.JSON_SCHEMA and not self.schema_:
            raise ValueError("JSON_SCHEMA 格式必须提供 schema")
        return self


class ChatUsage(WireModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    source: UsageSource = UsageSource.PROVIDER

    @model_validator(mode="after")
    def _fill_total(self) -> "ChatUsage":
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            object.__setattr__(
                self, "total_tokens", self.input_tokens + self.output_tokens
            )
        return self


class ChatModelRequest(WireModel):
    model: str
    messages: list[ModelMessage]
    tools: list[ModelTool] = Field(default_factory=list)
    tool_choice: ModelToolChoice = Field(default_factory=ModelToolChoice)
    options: ModelGenerationOptions = Field(default_factory=ModelGenerationOptions)
    response_format: ModelResponseFormat = Field(default_factory=ModelResponseFormat)
    stream: bool = False
    # 派生标志：是否续写最后一条 assistant；来源是消息层 is_prefix
    prefix: bool = False


class ChatModelResponse(WireModel):
    message: ModelMessage
    finish_reason: ChatFinishReason
    usage: ChatUsage = Field(default_factory=ChatUsage)
    response_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ——————————————————— 流式事件 ———————————————————

class _StreamEvent(WireModel):
    type: str


class ChatStreamStart(_StreamEvent):
    type: Literal["start"] = "start"
    model: str
    response_id: str | None = None


class ChatTextDelta(_StreamEvent):
    type: Literal["text_delta"] = "text_delta"
    text: str
    index: int = 0


class ChatThinkingDelta(_StreamEvent):
    type: Literal["thinking_delta"] = "thinking_delta"
    text: str = ""
    signature: str | None = None
    redacted: bool = False
    index: int = 0


class ChatToolCallStart(_StreamEvent):
    type: Literal["tool_call_start"] = "tool_call_start"
    call_id: str
    name: str
    index: int = 0


class ChatToolCallDelta(_StreamEvent):
    type: Literal["tool_call_delta"] = "tool_call_delta"
    call_id: str
    arguments_delta: str
    index: int = 0


class ChatToolCallEnd(_StreamEvent):
    type: Literal["tool_call_end"] = "tool_call_end"
    call_id: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None
    index: int = 0


class ChatUsageEvent(_StreamEvent):
    type: Literal["usage"] = "usage"
    usage: ChatUsage


class ChatStreamEnd(_StreamEvent):
    type: Literal["end"] = "end"
    finish_reason: ChatFinishReason
    usage: ChatUsage | None = None
    response_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatStreamError(_StreamEvent):
    model_config = ConfigDict(
        extra="forbid", frozen=True, arbitrary_types_allowed=True
    )

    type: Literal["error"] = "error"
    error: ProviderError


ChatStreamEvent = (
    ChatStreamStart
    | ChatTextDelta
    | ChatThinkingDelta
    | ChatToolCallStart
    | ChatToolCallDelta
    | ChatToolCallEnd
    | ChatUsageEvent
    | ChatStreamEnd
    | ChatStreamError
)
