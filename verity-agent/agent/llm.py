"""LLM 应用层公共类型（单文件版）。

消息模块的 ``GenerationInfo``、Hooks 的 ``LLMUsage``、Agent 的
``LLMRequest/LLMResponse`` 都从这里引用，不重复定义字段。
ChatService/ModelRegistry/PromptBuilder/Streaming 尚未实现；未来真正需要
服务实现时再拆分本模块为包，不提前造空文件。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config.llm import LLMConfig, StructuredOutputConfig
from exception.llm import LLMError, LLMErrorCode
from provider.base import ProviderModelSpec
from provider.enums import ResponseFormatType, ToolChoiceMode, UsageSource
from utils.time import utcnow

__all__ = [
    "LLMConfig",
    "StructuredOutputConfig",
    "LLMError",
    "LLMErrorCode",
    "ModelKind",
    "ModelPurpose",
    "ProviderModelRef",
    "ModelPricing",
    "ModelSpec",
    "ModelSelectionPolicy",
    "ModelSelection",
    "LLMRole",
    "FinishReason",
    "LLMUsage",
    "GenerationInfo",
    "TextPart",
    "ImagePart",
    "FilePart",
    "ThinkingPart",
    "LLMContentPart",
    "LLMToolCall",
    "LLMMessage",
    "LLMToolDescriptor",
    "ToolChoice",
    "GenerationOptions",
    "ResponseFormat",
    "LLMRequestContext",
    "LLMRequest",
    "LLMResponse",
    "LLMStreamPhase",
]


class ModelKind(StrEnum):
    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelPurpose(StrEnum):
    CHAT = "chat"
    COMPRESSION = "compression"
    MEMORY_EXTRACTION = "memory_extraction"
    TITLE = "title"
    # 文档级描述提取/精炼（RAG Indexer 使用）
    DOCUMENT_DESCRIPTION = "document_description"


class LLMRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    CANCELLED = "cancelled"


class LLMStreamPhase(StrEnum):
    PRE_STREAM = "pre_stream"
    STREAMING_NO_OUTPUT = "streaming_no_output"
    STREAMING_OUTPUT = "streaming_output"
    COMPLETED = "completed"
    FAILED = "failed"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderModelRef(_Model):
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)


class ModelPricing(_Model):
    provider: str
    model: str
    input_per_million: float = 0.0
    output_per_million: float = 0.0
    cached_input_per_million: float | None = None
    currency: str = "USD"
    effective_from: datetime = Field(default_factory=utcnow)


class ModelSpec(_Model):
    id: str = Field(min_length=1)
    kind: ModelKind
    provider_model: ProviderModelRef
    purposes: set[ModelPurpose] = Field(default_factory=set)
    defaults: GenerationOptions | None = None
    fallbacks: list[str] = Field(default_factory=list)
    pricing: ModelPricing | None = None
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelSelectionPolicy(_Model):
    purpose: ModelPurpose
    primary_model_id: str
    fallback_model_ids: list[str] = Field(default_factory=list)
    required_capabilities: set[str] = Field(default_factory=set)


class ModelSelection(_Model):
    model: ModelSpec
    provider_model: ProviderModelSpec
    fallbacks: list[ModelSpec] = Field(default_factory=list)
    reason: str | None = None


class LLMUsage(_Model):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    source: UsageSource = UsageSource.PROVIDER

    @model_validator(mode="after")
    def _fill_total(self) -> "LLMUsage":
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            object.__setattr__(
                self, "total_tokens", self.input_tokens + self.output_tokens
            )
        return self


class GenerationInfo(_Model):
    provider: str
    model: str
    request_id: str | None = None
    response_id: str | None = None
    finish_reason: FinishReason
    usage: LLMUsage | None = None
    warnings: list[str] = Field(default_factory=list)
    latency_ms: int | None = None


class TextPart(_Model):
    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    type: Literal["image"] = "image"
    url: str | None = None
    data: bytes | None = None
    file_id: UUID | None = None
    mime_type: str | None = None
    detail: str | None = None


class FilePart(_Model):
    type: Literal["file"] = "file"
    file_id: UUID | None = None
    url: str | None = None
    filename: str | None = None
    mime_type: str | None = None
    text: str | None = None


class ThinkingPart(_Model):
    type: Literal["thinking"] = "thinking"
    text: str = ""
    signature: str | None = None
    redacted: bool = False


LLMContentPart = TextPart | ImagePart | FilePart | ThinkingPart


class LLMToolCall(_Model):
    call_id: str
    name: str
    arguments: dict[str, Any] | None = None
    raw_arguments: str | None = None
    index: int | None = None


class LLMMessage(_Model):
    role: LLMRole
    content: list[LLMContentPart] = Field(default_factory=list)
    name: str | None = None
    tool_calls: list[LLMToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    prefix: bool = False
    cacheable: bool = False


class LLMToolDescriptor(_Model):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool = False
    version: str | None = None


class ToolChoice(_Model):
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    tool_name: str | None = None


class GenerationOptions(_Model):
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


class ResponseFormat(_Model):
    type: ResponseFormatType = ResponseFormatType.TEXT
    name: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class LLMRequestContext(_Model):
    trace_id: str
    conversation_id: UUID | None = None
    turn_id: str | None = None
    user_id: UUID | None = None
    call_id: str | None = None
    purpose: ModelPurpose = ModelPurpose.CHAT
    deadline: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(_Model):
    request_id: str
    context: LLMRequestContext
    # 已选应用模型 id；实际模型以 ModelSelection 为准（D-40）
    model_id: str
    messages: list[LLMMessage]
    tools: list[LLMToolDescriptor] = Field(default_factory=list)
    tool_choice: ToolChoice = Field(default_factory=ToolChoice)
    options: GenerationOptions = Field(default_factory=GenerationOptions)
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)
    stream: bool = False
    prefix_completion: bool = False


class LLMResponse(_Model):
    request_id: str
    provider: str
    model: str
    message: LLMMessage
    finish_reason: FinishReason
    usage: LLMUsage = Field(default_factory=LLMUsage)
    response_id: str | None = None
    latency_ms: int | None = None
    first_token_ms: int | None = None
    partial: bool = False
    phase: LLMStreamPhase | None = None
    warnings: list[str] = Field(default_factory=list)
