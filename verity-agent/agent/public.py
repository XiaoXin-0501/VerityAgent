"""Agent 公共 DTO。

外部只依赖这里的类型；Run/Turn/Step/Snapshot/Port 等运行态对象不导出。
消息、工具、LLM 的既有 DTO 直接复用，避免重复造投影。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent.llm import FinishReason, LLMUsage, ToolChoice
from agent.message.enum import MessageStatus
from agent.message.models import (
    AssistantMessage,
    FileRef,
    ImageRef,
    Thinking,
    ToolCall,
)
from rag.shared.enums import RAGMode
from utils.ids import new_request_id
from utils.time import utcnow

from .runtime.enums import (
    AgentErrorCode,
    AgentEventType,
    AgentWarningCode,
    CancelReason,
    RunStatus,
    TurnStatus,
)

__all__ = [
    "AgentRequest",
    "AgentRunRequest",
    "AgentResult",
    "AgentEvent",
    "AgentRunView",
    "AgentError",
    "AgentWarning",
    "FinishReason",
    "RAGMode",
    "CancelReason",
    "RunStatus",
    "TurnStatus",
    "AgentEventType",
    "AgentErrorCode",
    "AgentWarningCode",
]


class _Model(BaseModel):
    """公共 DTO 基类：拒绝未知字段，保持不可变。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class AgentRequest(_Model):
    """一次 Agent 请求；HTTP 层只做投影，不复用 Request 对象。"""

    # 请求链路 id；为空由 Agent 生成
    request_id: str = Field(default_factory=new_request_id)
    # 用户/会话归属
    user_id: UUID
    conversation_id: UUID
    # 消息幂等键；(conversation_id, client_message_id) 唯一
    client_message_id: UUID
    # 用户文本；可空串（仅附件）
    content: str = ""
    # 图片/会话文件引用
    images: list[ImageRef] = Field(default_factory=list)
    files: list[FileRef] = Field(default_factory=list)
    # 执行时权限快照；工具选择和 ToolContext 使用
    permissions: frozenset[str] = frozenset()
    # 是否流式；为空使用 AgentConfig.stream 默认（当前直接请求级覆盖）
    stream: bool | None = None
    # 覆盖主模型；为空使用 AgentDependencies.model_id
    model_id: str | None = None
    # RAG 模式；v1 framework 只保留字段，INJECTED 需要注入 RagPort
    rag_mode: RAGMode | None = None
    # 工具选择覆盖
    tool_choice: ToolChoice | None = None
    # 本次请求可收紧工具轮上限
    max_tool_rounds: int | None = Field(default=None, ge=1)
    # 客户端/实验信息；不进 Prompt
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("rag_mode", mode="before")
    @classmethod
    def _normalize_rag_mode(cls, value: object):
        """兼容 API 文档中 ``TOOL / INJECTED / OFF`` 的大写写法。"""
        if isinstance(value, str):
            return value.lower()
        return value

    @model_validator(mode="after")
    def _has_payload(self) -> "AgentRequest":
        """请求必须提供文本、图片或文件之一。"""
        if not self.content.strip() and not self.images and not self.files:
            raise ValueError("content/images/files 至少有一个非空")
        return self


# 文档中的内部别名；公共代码仍使用 AgentRequest。
AgentRunRequest = AgentRequest


class AgentError(_Model):
    """对外错误；message 必须已脱敏。"""

    code: AgentErrorCode
    message: str
    retryable: bool = False
    user_visible: bool = True
    details: dict[str, Any] = Field(default_factory=dict)
    cause: str | None = None
    partial_result: bool = False


class AgentWarning(_Model):
    """可降级告警；随 result/SSE 返回并写日志。"""

    code: AgentWarningCode
    message: str
    user_visible: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class AgentEvent(_Model):
    """SSE 事件信封；``id`` 由 run_id + seq 计算。"""

    seq: int = Field(ge=1)
    event_type: AgentEventType
    run_id: str
    turn_id: str
    message_id: UUID | None = None
    call_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def id(self) -> str:
        """SSE Last-Event-ID 游标。"""
        return f"{self.run_id}:{self.seq}"


class AgentResult(_Model):
    """一次 run 的终态结果；正常失败也通过该对象返回。"""

    run_id: str
    conversation_id: UUID
    turn_id: str
    status: RunStatus
    user_message_id: UUID
    assistant_message: AssistantMessage | None = None
    message_ids: list[UUID] = Field(default_factory=list)
    usage: LLMUsage = Field(default_factory=LLMUsage)
    warnings: list[AgentWarning] = Field(default_factory=list)
    error: AgentError | None = None
    duration_ms: int = 0


class AgentRunView(_Model):
    """只读运行视图；get_run/attach 共用。"""

    run_id: str
    turn_id: str
    conversation_id: UUID
    status: RunStatus
    turn_status: TurnStatus = TurnStatus.CREATED
    user_message_id: UUID | None = None
    assistant_message_id: UUID | None = None
    assistant_status: MessageStatus | None = None
    content: str = ""
    thinking: list[Thinking] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: LLMUsage | None = None
    last_event_id: str | None = None
    result: AgentResult | None = None
    error: AgentError | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
