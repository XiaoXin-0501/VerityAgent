"""消息领域模型。

结构：

- ``enum.py``   ：消息角色/状态/来源等枚举；
- ``models.py`` ：5 个消息类 + 图片/文件/工具/产物等支撑模型 + ``StoredMessage``；
- ``codec.py``  ：领域消息 ↔ ``conversation_messages.content`` JSONB 的编解码。

典型用法::

    from agent.message import UserMessage, ImageRef, encode_content

    message = UserMessage(
        conversation_id=conversation_id,
        content="这张图里是什么？",
        images=[ImageRef(url="https://cdn.example.com/a.png", detail="high")],
    )
    payload = encode_content(message)   # 写入 conversation_messages.content
"""

from .codec import (
    MESSAGE_CONTENT_VERSION,
    decode_content,
    decode_row,
    decode_stored_row,
    encode_content,
)
from .enum import (
    ArtifactKind,
    ImageDetail,
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
    Message,
    MessageModel,
    PromptSection,
    StoredMessage,
    SystemMessage,
    Thinking,
    ToolCall,
    ToolFile,
    ToolImage,
    ToolMessage,
    Turn,
    UserMessage,
    utcnow,
)

__all__ = [
    # 编解码
    "MESSAGE_CONTENT_VERSION",
    "encode_content",
    "decode_content",
    "decode_row",
    "decode_stored_row",
    # 枚举
    "MessageRole",
    "MessageStatus",
    "SystemOrigin",
    "ImageDetail",
    "ToolCallStatus",
    "ArtifactKind",
    "TurnStatus",
    # 消息类
    "MessageModel",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "Message",
    # 支撑模型
    "ImageRef",
    "FileRef",
    "Thinking",
    "ToolCall",
    "Artifact",
    "ToolFile",
    "ToolImage",
    "StoredMessage",
    "Turn",
    "PromptSection",
    "utcnow",
]
