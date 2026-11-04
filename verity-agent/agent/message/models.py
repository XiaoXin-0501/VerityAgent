"""消息领域模型（扁平模型）。

与 ``doc/消息数据模型设计.md`` 对齐：

- ``BaseMessage`` 只保留 ``id / conversation_id / role / status / created_at / content``；
- System / User / Assistant / Tool 的角色私有字段直接挂在子类，不再嵌套 metadata；
- ``seq_no`` 不在这里——仓库层用 :class:`StoredMessage` 携带；
- ``content`` 永远是文本；图片、文件、思考、工具、产物都是旁路字段。

模型只负责“字段 + 类型/业务校验”，存储 JSON 的编解码见 ``codec.py``。
"""

from __future__ import annotations

import json as jsonlib
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enum import (
    ArtifactKind,
    ImageDetail,
    MessageRole,
    MessageStatus,
    SystemOrigin,
    ToolCallStatus,
    TurnStatus,
)


def utcnow() -> datetime:
    """带时区的 UTC 当前时间。"""
    return datetime.now(timezone.utc)


class MessageModel(BaseModel):
    """消息相关模型的公共配置。"""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    def to_json_dict(self) -> dict[str, Any]:
        """转成可直接写入 JSONB 的字典（别名 + JSON 原生类型）。"""
        return self.model_dump(mode="json", by_alias=True)


class BaseMessage(MessageModel):
    """所有消息的公共字段。

    ``role`` 由子类收窄为唯一字面量；``content`` 是文本正文。
    没有 ``seq_no``：它是数据库行/压缩批次的事，由 ``StoredMessage`` 携带。
    """

    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    role: MessageRole
    status: MessageStatus = MessageStatus.COMPLETED
    created_at: datetime = Field(default_factory=utcnow)
    content: str = ""


# ——————————————————— 用户消息的支撑模型 ———————————————————


class ImageRef(MessageModel):
    """用户图片引用。

    只存处理器给出的稳定 URL；不存 base64 / object_key / mime / size。
    """

    url: str = Field(min_length=1)
    detail: ImageDetail = ImageDetail.AUTO

    @field_validator("url")
    @classmethod
    def _strip_url(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ImageRef.url 不能为空")
        return value


class FileRef(MessageModel):
    """用户文件引用。

    只存 ``session_files.id``；解析产物、状态、大小都在服务端。
    """

    file_id: UUID
    filename: str | None = Field(default=None, min_length=1)


# ——————————————————— 助手消息的支撑模型 ———————————————————


class Thinking(MessageModel):
    """思考/推理内容。``signature`` 供 Anthropic thinking 回传。"""

    text: str = Field(min_length=1)
    signature: str | None = Field(default=None, min_length=1)


class ToolCall(MessageModel):
    """LLM 发起的一次工具调用。"""

    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING


class Artifact(MessageModel):
    """助手生成的图片/文件/图表等产物。"""

    kind: ArtifactKind
    url: str = Field(min_length=1)
    filename: str | None = Field(default=None, min_length=1)


# ——————————————————— 工具消息的支撑模型 ———————————————————


class ToolFile(MessageModel):
    """工具返回/生成的文件。"""

    filename: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1)
    content: str | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "ToolFile":
        if self.url is None and self.content is None:
            raise ValueError("ToolFile 至少需要 url 或 content")
        return self


class ToolImage(MessageModel):
    """工具返回/生成的图片。"""

    url: str = Field(min_length=1)
    detail: ImageDetail | None = None


# ——————————————————— 四个消息子类 ———————————————————


class SystemMessage(BaseMessage):
    """配置型系统指令。

    动态的长期记忆/会话摘要/RAG 上下文是 ``PromptSection``，不是本类。
    """

    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM
    name: str = Field(default="default", min_length=1)
    origin: SystemOrigin = SystemOrigin.BASE

    @model_validator(mode="after")
    def _validate_content(self) -> "SystemMessage":
        if not self.content.strip():
            raise ValueError("SystemMessage.content 不能为空")
        return self


class UserMessage(BaseMessage):
    """用户消息：文本 + 图片 + 文件。"""

    role: Literal[MessageRole.USER] = MessageRole.USER
    images: list[ImageRef] = Field(default_factory=list)
    files: list[FileRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_payload(self) -> "UserMessage":
        if not self.content.strip() and not self.images and not self.files:
            raise ValueError("UserMessage 至少需要 content、images、files 之一")
        file_ids = [file.file_id for file in self.files]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("UserMessage.files 中 file_id 不能重复")
        return self


class AssistantMessage(BaseMessage):
    """助手消息：文本 + 思考 + 工具调用 + 生成产物 + 前缀续写标记。"""

    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    thinking: Thinking | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    is_prefix: bool = False

    @model_validator(mode="after")
    def _validate_payload(self) -> "AssistantMessage":
        if self.status in (MessageStatus.COMPLETED, MessageStatus.PARTIAL):
            if (
                not self.content.strip()
                and self.thinking is None
                and not self.tool_calls
                and not self.artifacts
            ):
                raise ValueError(
                    "AssistantMessage 至少需要 content、thinking、tool_calls、artifacts 之一"
                )
        call_ids = [call.call_id for call in self.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise ValueError("同一 AssistantMessage 内 call_id 不能重复")
        return self


class ToolMessage(BaseMessage):
    """工具消息：成功文本或错误信息 + 结构化结果/文件/图片。"""

    role: Literal[MessageRole.TOOL] = MessageRole.TOOL
    tool_call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    # 存储协议里的 key 是 ``json``；Python 属性用 json_data，避免与 BaseModel 冲突。
    json_data: Any | None = Field(default=None, alias="json")
    files: list[ToolFile] = Field(default_factory=list)
    images: list[ToolImage] = Field(default_factory=list)

    @field_validator("json_data")
    @classmethod
    def _validate_json_serializable(cls, value: Any) -> Any:
        if value is None:
            return value
        try:
            jsonlib.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("ToolMessage.json 必须是可 JSON 序列化的数据") from exc
        return value

    @model_validator(mode="after")
    def _validate_payload(self) -> "ToolMessage":
        if (
            not self.content.strip()
            and self.json_data is None
            and not self.files
            and not self.images
        ):
            raise ValueError(
                "ToolMessage 至少需要 content、json、files、images 之一"
            )
        return self


# ——————————————————— 消息联合类型与运行时包装 ———————————————————

Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role"),
]


class StoredMessage(MessageModel):
    """数据库行视图：seq_no 是存储排序号，不属于领域消息。"""

    seq_no: int = Field(gt=0)
    message: Message


class Turn(MessageModel):
    """一次用户回合的聚合视图（不是消息类）。"""

    conversation_id: UUID
    ordinal: int = Field(ge=1)
    user_message_id: UUID
    message_ids: list[UUID] = Field(default_factory=list)
    status: TurnStatus = TurnStatus.RUNNING
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_time_range(self) -> "Turn":
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("Turn.ended_at 不能早于 started_at")
        return self


class PromptSection(MessageModel):
    """每轮动态拼装的提示词片段（长期记忆/摘要/RAG/工具策略）。

    不是消息，不落 ``conversation_messages``。
    """

    name: str = Field(min_length=1)
    content: str = Field(min_length=1)


__all__ = [
    "MessageModel",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "ImageRef",
    "FileRef",
    "Thinking",
    "ToolCall",
    "Artifact",
    "ToolFile",
    "ToolImage",
    "Message",
    "StoredMessage",
    "Turn",
    "PromptSection",
    "utcnow",
]
