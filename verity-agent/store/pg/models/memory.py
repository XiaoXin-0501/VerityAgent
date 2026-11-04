"""user_memory / user_messages 行模型，以及长期记忆画像 JSON 协议。

``user_memory.value`` 固定 7 段（现在时 4 段 + 过去时 3 桶），每段自带
``summary`` 与 ``updatedAt``。字段在 Python 侧用 snake_case，落库时通过
``by_alias`` 序列化成文档约定的 camelCase。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping
from uuid import UUID

from pydantic import Field

from .base import DBModel, DBJsonModel, utcnow
from .conversation import MessageContent
from .enums import UserMessageStatus


class MemorySection(DBJsonModel):
    summary: str = ""
    updated_at: datetime = Field(default_factory=utcnow, alias="updatedAt")


class MemoryUserSections(DBJsonModel):
    """现在时 4 段：用户当下是什么状态。"""

    work_context: MemorySection = Field(
        default_factory=MemorySection, alias="workContext"
    )
    personal_context: MemorySection = Field(
        default_factory=MemorySection, alias="personalContext"
    )
    top_of_mind: MemorySection = Field(
        default_factory=MemorySection, alias="topOfMind"
    )
    cognitive_style: MemorySection = Field(
        default_factory=MemorySection, alias="cognitiveStyle"
    )


class MemoryHistorySections(DBJsonModel):
    """过去时 3 桶：按时间远近分桶的历史背景。"""

    recent_months: MemorySection = Field(
        default_factory=MemorySection, alias="recentMonths"
    )
    earlier_context: MemorySection = Field(
        default_factory=MemorySection, alias="earlierContext"
    )
    long_term_background: MemorySection = Field(
        default_factory=MemorySection, alias="longTermBackground"
    )


class MemoryValue(DBJsonModel):
    """``user_memory.value`` 的固定结构（画像摘要，不放原子事实）。"""

    last_updated: datetime = Field(default_factory=utcnow, alias="lastUpdated")
    user: MemoryUserSections = Field(default_factory=MemoryUserSections)
    history: MemoryHistorySections = Field(default_factory=MemoryHistorySections)


class UserMemory(DBModel):
    id: UUID
    user_id: UUID
    value: MemoryValue
    revision: int
    fact_count: int
    updated_at: datetime


class UserMessage(DBModel):
    """从 conversation_messages 同步双写出来的暂存行，供长期记忆 worker 扫描。"""

    id: UUID
    user_id: UUID
    conversation_id: UUID
    message_id: UUID
    content: MessageContent
    status: UserMessageStatus
    created_at: datetime


@dataclass(slots=True)
class MemoryChange:
    """``UserMemoryRepository.update_with_retry`` 的回调返回值。

    字段为 ``None`` 表示"本次不修改该列"；两列都是 ``None`` 时视为无变更。
    """

    value: MemoryValue | Mapping[str, Any] | None = None
    fact_count: int | None = None


def empty_memory_value() -> MemoryValue:
    """新建用户时初始化一份空画像。"""
    return MemoryValue()


__all__ = [
    "MemorySection",
    "MemoryUserSections",
    "MemoryHistorySections",
    "MemoryValue",
    "MemoryChange",
    "UserMemory",
    "UserMessage",
    "empty_memory_value",
]

