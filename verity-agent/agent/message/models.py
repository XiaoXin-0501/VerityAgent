"""消息领域模型 v2（《消息数据模型设计》§4–§6）。

- 基类只放公共字段，角色私有字段挂在子类；
- ``content`` 是文本正文；多模态/思考/工具/产物都是旁路字段；
- ``seq_no`` 只在存储层（``StoredMessage``），不进领域消息；
- ``GenerationInfo`` 的权威定义在 ``agent.llm.types``，这里只引用。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.llm.types import GenerationInfo
from utils.ids import uuid7
from utils.serialization import to_jsonable
from utils.time import utcnow

from .enum import (
    ArtifactKind,
    ImageDetail,
    MessageRole,
    MessageStatus,
    SystemOrigin,
    ToolCallStatus,
    TurnStatus,
)

__all__ = [
    "BaseMessage",
    "SystemMessage",
    "ImageRef",
    "FileRef",
    "UserMessage",
    "Thinking",
    "ToolCall",
    "Artifact",
    "AssistantMessage",
    "ToolFile",
    "ToolImage",
    "ToolMessage",
    "RawMessage",
    "StoredMessage",
    "Turn",
]


class MessageModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        validate_assignment=True,
    )


class BaseMessage(MessageModel):
    id: UUID = Field(default_factory=uuid7)
    conversation_id: UUID
    role: MessageRole
    status: MessageStatus = MessageStatus.COMPLETED
    created_at: datetime = Field(default_factory=utcnow)
    content: str = ""


class SystemMessage(BaseMessage):
    """只承载配置型 system prompt；动态上下文由 ContextSection 组装。"""

    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM  # pyright: ignore[reportIncompatibleVariableOverride]
    name: str = "default"
    origin: SystemOrigin = SystemOrigin.BASE

    @model_validator(mode="after")
    def _completed_only(self) -> Self:
        if self.status is not MessageStatus.COMPLETED:
            raise ValueError("SystemMessage 写入即 completed")
        return self


class ImageRef(MessageModel):
    url: str
    detail: ImageDetail = ImageDetail.AUTO

    @model_validator(mode="after")
    def _url_required(self) -> Self:
        if not self.url.strip():
            raise ValueError("ImageRef.url 不能为空")
        return self


class FileRef(MessageModel):
    file_id: UUID
    filename: str | None = None


class UserMessage(BaseMessage):
    role: Literal[MessageRole.USER] = MessageRole.USER  # pyright: ignore[reportIncompatibleVariableOverride]
    images: list[ImageRef] = Field(default_factory=list)
    files: list[FileRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_payload(self) -> Self:
        if not self.content.strip() and not self.images and not self.files:
            raise ValueError("用户消息的 content/images/files 至少有一个非空")
        return self

    @model_validator(mode="after")
    def _unique_files(self) -> Self:
        ids = [item.file_id for item in self.files]
        if len(ids) != len(set(ids)):
            raise ValueError("同一用户消息的 file_id 不能重复")
        return self

    @model_validator(mode="after")
    def _completed_only(self) -> Self:
        if self.status is not MessageStatus.COMPLETED:
            raise ValueError("UserMessage 写入即 completed")
        return self


class Thinking(MessageModel):
    text: str = ""
    signature: str | None = None
    redacted: bool = False

    @model_validator(mode="after")
    def _has_content(self) -> Self:
        if not self.redacted and not self.text:
            raise ValueError("非 redacted 的 thinking 必须有文本")
        return self


class ToolCall(MessageModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    raw_arguments: str | None = None
    status: ToolCallStatus = ToolCallStatus.PENDING

    @model_validator(mode="after")
    def _ids_required(self) -> Self:
        if not self.call_id.strip():
            raise ValueError("ToolCall.call_id 不能为空")
        if not self.name.strip():
            raise ValueError("ToolCall.name 不能为空")
        return self


class Artifact(MessageModel):
    kind: ArtifactKind
    url: str
    filename: str | None = None

    @model_validator(mode="after")
    def _url_required(self) -> Self:
        if not self.url.strip():
            raise ValueError("Artifact.url 不能为空")
        return self


class AssistantMessage(BaseMessage):
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT  # pyright: ignore[reportIncompatibleVariableOverride]
    thinking: list[Thinking] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    is_prefix: bool = False
    generation: GenerationInfo | None = None

    @model_validator(mode="after")
    def _unique_call_ids(self) -> Self:
        ids = [call.call_id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("同一 assistant 消息的 call_id 不能重复")
        return self

    @model_validator(mode="after")
    def _has_payload(self) -> Self:
        if self.status in (MessageStatus.COMPLETED, MessageStatus.PARTIAL):
            if not (
                self.content.strip()
                or self.thinking
                or self.tool_calls
                or self.artifacts
            ):
                raise ValueError(
                    "完成的 assistant 消息 content/thinking/tool_calls/artifacts"
                    " 至少有一个非空"
                )
        return self


class ToolFile(MessageModel):
    filename: str | None = None
    url: str | None = None
    content: str | None = None

    @model_validator(mode="after")
    def _has_source(self) -> Self:
        if not self.url and not self.content:
            raise ValueError("ToolFile 必须提供 url 或 content")
        return self


class ToolImage(MessageModel):
    url: str
    detail: ImageDetail | None = None

    @model_validator(mode="after")
    def _url_required(self) -> Self:
        if not self.url.strip():
            raise ValueError("ToolImage.url 不能为空")
        return self


class ToolMessage(BaseMessage):
    role: Literal[MessageRole.TOOL] = MessageRole.TOOL  # pyright: ignore[reportIncompatibleVariableOverride]
    tool_call_id: str
    name: str
    json_data: Any = Field(default=None, alias="json")
    files: list[ToolFile] = Field(default_factory=list)
    images: list[ToolImage] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_result(self) -> Self:
        if not (
            self.content.strip() or self.json_data is not None or self.files or self.images
        ):
            raise ValueError("ToolMessage 必须提供 content/json/files/images 之一")
        return self

    @model_validator(mode="after")
    def _tool_status(self) -> Self:
        if self.status not in (MessageStatus.COMPLETED, MessageStatus.FAILED):
            raise ValueError("ToolMessage 只能是 completed 或 failed")
        return self

    @model_validator(mode="after")
    def _json_serializable(self) -> Self:
        if self.json_data is not None:
            try:
                to_jsonable(self.json_data)
            except TypeError as exc:
                raise ValueError("ToolMessage.json 必须可 JSON 序列化") from exc
        return self


class RawMessage(BaseModel):
    """未知版本/解析失败时的只读降级消息；不进入 LLM/压缩/抽取。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: UUID = Field(default_factory=uuid7)
    conversation_id: UUID
    # 保留原始 role；未知 role 也要能展示/排障
    role: MessageRole | str
    status: MessageStatus
    created_at: datetime = Field(default_factory=utcnow)
    content: dict[str, Any] = Field(default_factory=dict)
    v: int | None = None
    parse_error: str | None = None


class StoredMessage(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    seq_no: int = Field(gt=0)
    message: BaseMessage | RawMessage


class Turn(BaseModel):
    """只读回合投影，用于 UI 分组/压缩边界；运行态用 AgentTurn。"""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    conversation_id: UUID
    ordinal: int = Field(ge=1)
    user_message_id: UUID
    message_ids: list[UUID] = Field(default_factory=list)
    status: TurnStatus = TurnStatus.RUNNING
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None


def is_message(value: BaseModel) -> bool:
    return isinstance(value, BaseMessage)
