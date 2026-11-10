"""Provider 层枚举。

Provider 层不依赖 LLM/Agent 的消息与工具模型，只使用这里的厂商无关枚举；
LLM 层的 ``ModelKind`` 等枚举与之一一映射（D-39）。
"""

from __future__ import annotations

from enum import StrEnum

# ProviderKind / ChatApiStyle 由配置包统一定义，这里只做领域层重新导出
from config.provider import ChatApiStyle, ProviderKind
from exception.provider import ProviderErrorCode

__all__ = [
    "ProviderKind",
    "ProviderModelKind",
    "ChatApiStyle",
    "ModelRole",
    "ToolChoiceMode",
    "ResponseFormatType",
    "ChatFinishReason",
    "UsageSource",
    "ProviderHealthStatus",
    "ProviderErrorCode",
]


class ProviderModelKind(StrEnum):
    """Provider 内模型类型（D-39）。"""

    CHAT = "chat"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolChoiceMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    TOOL = "tool"


class ResponseFormatType(StrEnum):
    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class ChatFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    CANCELLED = "cancelled"


class UsageSource(StrEnum):
    PROVIDER = "provider"
    ESTIMATED = "estimated"


class ProviderHealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"
