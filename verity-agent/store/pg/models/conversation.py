"""conversations / conversation_messages / conversation_summaries 行模型。

消息 ``content`` 的权威协议在 ``agent.message``（v2），本模块只把 JSONB
当作 ``dict`` 存储，避免在存储层维护第二套消息模型。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator

from .base import DBModel, DBJsonModel, utcnow
from .enums import ConversationStatus, MessageRole, MessageStatus, RunStatus

__all__ = [
    "Conversation",
    "ConversationMessage",
    "ConversationSummary",
    "SessionSummary",
    "SummaryTopic",
    "SummaryFact",
    "SummaryConstraint",
]


# ——————————————————— 会话摘要 summary_json 协议 ———————————————————

class SummaryTopic(DBJsonModel):
    # 主题 id；facts 通过 ``topic_id`` 关联。JSON 字段名为 ``id``，
    # 旧摘要的 ``topic_id`` 由兼容校验映射过来。
    id: UUID = Field(default_factory=uuid4)
    name: str
    summary: str
    # 支撑该主题的轮次 id（一次提问到完整回答为一轮）；v3 起由模型从输入
    # 消息的 ``[round:...]`` 标签中选择，供轮次召回工具回查原始问答。
    source_round_ids: list[UUID] = Field(default_factory=list)
    mention_count: int = Field(default=1, ge=1)
    last_mentioned_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy(cls, data: Any) -> Any:
        """兼容旧摘要：``topic_id`` → ``id``，并丢弃已废弃的 topic 消息锚点。"""
        if not isinstance(data, dict):
            return data
        value = dict(data)
        if "id" not in value and "topic_id" in value:
            value["id"] = value.pop("topic_id")
        value.pop("topic_id", None)
        value.pop("source_msg_ids", None)
        return value


class SummaryFact(DBJsonModel):
    fact_id: UUID = Field(default_factory=uuid4)
    content: str
    # 该事实归属的主题；模型只能引用本次输出中存在的 topic id
    topic_id: UUID | None = None
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
    """conversation_summaries.summary_json 的固定三层结构。

    v3 起：topic 只保留 ``source_round_ids``（轮次 id），facts 可带
    ``topic_id`` 关联主题；读取兼容 v1/v2（旧 ``topic_id`` 映射为 ``id``，
    旧 topic 消息锚点丢弃），新写入默认 v3。
    """

    version: Literal[1, 2, 3] = 3
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
    current_run_id: UUID | None = None
    run_started_at: datetime | None = None
    lock_expires_at: datetime | None = None
    turn_count: int
    total_input_tokens: int
    total_output_tokens: int
    created_at: datetime
    last_message_at: datetime | None = None
    deleted_at: datetime | None = None
    delete_attempts: int = 0
    delete_error: str | None = None
    delete_next_retry_at: datetime | None = None


class ConversationMessage(DBModel):
    id: UUID
    conversation_id: UUID
    seq_no: int
    client_message_id: UUID | None = None
    # 轮次 id：一次用户提问到完整回答的所有消息共享；旧数据可能为 NULL
    round_id: UUID | None = None
    role: MessageRole
    content: dict[str, Any]
    status: MessageStatus
    created_at: datetime
    updated_at: datetime


class ConversationSummary(DBModel):
    conversation_id: UUID
    summary_json: SessionSummary
    covered_through_seq: int
    compression_count: int = 0
    updated_at: datetime
