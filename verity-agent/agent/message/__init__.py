"""消息领域模型 v2 与存储编解码。"""

from .codec import (
    MESSAGE_CONTENT_VERSION,
    content_version,
    decode_content,
    decode_row,
    decode_stored_row,
    encode_content,
)
from config.message import MessageConfig
from .enum import (
    ArtifactKind,
    ImageDetail,
    MessageContentVersion,
    MessageRole,
    MessageStatus,
    SystemOrigin,
    ToolCallStatus,
    TurnStatus,
)
from .models import (
    Artifact,
    AssistantMessage,
    BaseMessage,
    FileRef,
    ImageRef,
    RawMessage,
    StoredMessage,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolFile,
    ToolImage,
    ToolMessage,
    Turn,
    UserMessage,
)

__all__ = [
    # codec
    "MESSAGE_CONTENT_VERSION",
    "encode_content",
    "decode_content",
    "decode_row",
    "decode_stored_row",
    "content_version",
    # config
    "MessageConfig",
    # enums
    "MessageRole",
    "MessageStatus",
    "SystemOrigin",
    "ImageDetail",
    "ToolCallStatus",
    "ArtifactKind",
    "TurnStatus",
    "MessageContentVersion",
    # models
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "RawMessage",
    "StoredMessage",
    "Turn",
    "ImageRef",
    "FileRef",
    "Thinking",
    "ToolCall",
    "Artifact",
    "ToolFile",
    "ToolImage",
]
