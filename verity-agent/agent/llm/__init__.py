"""LLM 应用层：类型、错误与配置。

ChatService / ModelRegistry / PromptBuilder / Streaming 在后续实现中放到本包；
权威字段定义见《LLM 模块数据结构设计》。
"""

from config.llm import LLMConfig, StructuredOutputConfig
from exception.llm import LLMError, LLMErrorCode
from .types import (
    FilePart,
    FinishReason,
    GenerationInfo,
    GenerationOptions,
    ImagePart,
    LLMContentPart,
    LLMMessage,
    LLMRequest,
    LLMRequestContext,
    LLMResponse,
    LLMRole,
    LLMStreamPhase,
    LLMToolCall,
    LLMToolDescriptor,
    LLMUsage,
    ModelKind,
    ModelPricing,
    ModelPurpose,
    ModelSelection,
    ModelSelectionPolicy,
    ModelSpec,
    ProviderModelRef,
    ResponseFormat,
    TextPart,
    ThinkingPart,
    ToolChoice,
)

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
