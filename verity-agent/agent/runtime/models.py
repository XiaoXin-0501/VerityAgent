"""Agent runtime 内部模型；不进入公共导出。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from agent.message.models import ToolCall
from utils.time import utcnow

from .enums import (
    AgentAction,
    AgentDecisionReason,
    AgentWarningCode,
    ToolRoundStatus,
)

__all__ = [
    "AgentLimits",
    "AgentDecision",
    "AgentToolRound",
    "AgentRunState",
    "AgentWarningData",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)


class AgentWarningData(_Model):
    """内部告警数据；由 loop 转成公共 AgentWarning。"""

    code: AgentWarningCode
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentLimits(_Model):
    """一次 run 派生出的生效限制。"""

    deadline: datetime
    max_tool_rounds: int
    max_llm_calls: int
    max_parallel_tools: int


class AgentDecision(_Model):
    """LLM 返回后的下一步决策（目标契约；framework v1 仍按 finish_reason 分支）。"""

    action: AgentAction
    reason: AgentDecisionReason
    tool_calls: list[ToolCall] = Field(default_factory=list)
    warnings: list[AgentWarningData] = Field(default_factory=list)


class AgentToolRound(_Model):
    """一轮工具调用（目标 trace 模型；framework v1 尚未落库/使用）。"""

    round_no: int
    assistant_message_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_message_ids: list[str] = Field(default_factory=list)
    status: ToolRoundStatus = ToolRoundStatus.PENDING
    router_activated: bool = False
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None


class AgentRunState(_Model):
    """运行期最小状态（目标契约；framework v1 使用 facade._ActiveRun）。"""

    run_id: str
    turn_id: str
    conversation_id: UUID
    status: str = "running"
    cancel_requested: bool = False
    last_event_id: str | None = None
    updated_at: datetime = Field(default_factory=utcnow)
