"""消息领域枚举。

这些值必须与 ``store.pg.models.enums`` 中同名枚举保持一致（数据库 CHECK
约束以那里为准）。领域层单独定义一份，避免 ``agent`` 反向依赖 ``store``；
以后如需统一，可让存储层改为 import 这里。
"""

from __future__ import annotations

from enum import StrEnum


class MessageRole(StrEnum):
    """会话消息角色，与 ``conversation_messages.role`` 对齐。"""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


class MessageStatus(StrEnum):
    """单条消息的内容生成状态，与 ``conversation_messages.status`` 对齐。"""

    STREAMING = "streaming"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class SystemOrigin(StrEnum):
    """SystemMessage 的来源，用于 Prompt 组装与审计。"""

    BASE = "base"
    CONFIGURED = "configured"
    TOOL_POLICY = "tool_policy"


class ImageDetail(StrEnum):
    """图片分辨率提示，透传给支持该能力的模型。"""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"


class ToolCallStatus(StrEnum):
    """单次工具调用的执行状态，不是消息状态。"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactKind(StrEnum):
    """助手/工具生成产物的类型。"""

    IMAGE = "image"
    FILE = "file"
    CHART = "chart"
    REPORT = "report"
    OTHER = "other"


class TurnStatus(StrEnum):
    """一次用户回合的整体状态。"""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


__all__ = [
    "MessageRole",
    "MessageStatus",
    "SystemOrigin",
    "ImageDetail",
    "ToolCallStatus",
    "ArtifactKind",
    "TurnStatus",
]
