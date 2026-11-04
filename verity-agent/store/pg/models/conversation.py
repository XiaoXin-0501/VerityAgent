"""conversations / conversation_messages / conversation_summaries 行模型。

另外定义了消息 content 与摘要 summary_json 的固定 JSON 协议：

- ``MessageContent``：``type = text / tool_call / tool_result / multimodal``；
- ``SessionSummary``：``topics`` / ``facts`` / ``constraints`` 三个独立顶层数组，
  事实与约束不嵌在话题里（见《Agent-RAG 系统设计方案》§2.3.3）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import ConfigDict, Field

from .base import DBModel, DBJsonModel, utcnow
from .enums import ConversationStatus, MessageRole, MessageStatus, RunStatus


# ——————————————————— 消息 content 协议 ———————————————————

class ToolCall(DBJsonModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class MessageContent(DBJsonModel):
    """conversation_messages.content 的固定协议。

    通过 ``extra="allow"`` 允许工具在协议外层附加业务字段，但 ``type`` 必须
    在约定范围内，避免各工具返回自定形态、下游无法解析。
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: Literal["text", "tool_call", "tool_result", "multimodal"] = "text"
    text: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    name: str | None = None
    result: Any = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)


# ——————————————————— 会话摘要 summary_json 协议 ———————————————————

class SummaryTopic(DBJsonModel):
    topic_id: UUID = Field(default_factory=uuid4)
    name: str
    summary: str
    mention_count: int = Field(default=1, ge=1)
    last_mentioned_at: datetime = Field(default_factory=utcnow)


class SummaryFact(DBJsonModel):
    fact_id: UUID = Field(default_factory=uuid4)
    content: str
    # 指向 conversation_messages.id，下次压缩时回喂校验，防错误累积
    source_msg_ids: list[UUID] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confirmation_count: int = Field(default=1, ge=1)
    last_confirmed_at: datetime = Field(default_factory=utcnow)


class SummaryConstraint(DBJsonModel):
    constraint_id: UUID = Field(default_factory=uuid4)
    content: str
    source_msg_ids: list[UUID] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    confirmation_count: int = Field(default=1, ge=1)
    last_confirmed_at: datetime = Field(default_factory=utcnow)


class SessionSummary(DBJsonModel):
    """conversation_summaries.summary_json 的固定三层结构。"""

    version: Literal[1] = 1
    topics: list[SummaryTopic] = Field(default_factory=list)
    facts: list[SummaryFact] = Field(default_factory=list)
    constraints: list[SummaryConstraint] = Field(default_factory=list)


# ——————————————————— 行模型 ———————————————————

class Conversation(DBModel):
    id: UUID
    user_id: UUID
    title: str
    status: ConversationStatus
    current_run_status: RunStatus
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    created_at: datetime
    last_message_at: datetime | None = None


class ConversationMessage(DBModel):
    id: UUID
    conversation_id: UUID
    seq_no: int
    role: MessageRole
    content: dict[str, Any]
    status: MessageStatus
    created_at: datetime


class ConversationSummary(DBModel):
    conversation_id: UUID
    summary_json: SessionSummary
    covered_through_seq: int
    updated_at: datetime
