"""消息领域枚举（v2）。"""

from __future__ import annotations

from enum import IntEnum, StrEnum

__all__ = [
    "MessageRole",
    "MessageStatus",
    "SystemOrigin",
    "ImageDetail",
    "ToolCallStatus",
    "ArtifactKind",
    "TurnStatus",
    "MessageContentVersion",
]


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class MessageStatus(StrEnum):
    STREAMING = "streaming"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class SystemOrigin(StrEnum):
    """SystemMessage 只承载配置型 prompt；动态上下文走 ContextSection。"""

    BASE = "base"
    CONFIGURED = "configured"


class ImageDetail(StrEnum):
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class ToolCallStatus(StrEnum):
    """运行期快照；持久化事实以 ToolMessage 为准。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    IMAGE = "image"
    FILE = "file"
    CHART = "chart"
    REPORT = "report"
    OTHER = "other"


class TurnStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class MessageContentVersion(IntEnum):
    """content.v；写入只写 v2，读取兼容 v1。"""

    V1 = 1
    V2 = 2
