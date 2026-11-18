"""用户长期画像读取适配器。"""

from __future__ import annotations

from uuid import UUID

from store.pg.models.memory import MemoryValue
from store.pg.models.conversation import SessionSummary
from store.pg.repositories import (
    ConversationSummaryRepository,
    UserMemoryRepository,
)
from utils.tokens import truncate_to_tokens

__all__ = ["PgMemoryProfileAdapter"]


class PgMemoryProfileAdapter:
    """从 ``user_memory`` / ``conversation_summaries`` 读取可注入文本。"""

    def __init__(
        self,
        repository: UserMemoryRepository | None = None,
        summary_repository: ConversationSummaryRepository | None = None,
    ) -> None:
        """保存仓库；不主动建立连接。"""
        self.repository = repository or UserMemoryRepository()
        self.summary_repository = (
            summary_repository or ConversationSummaryRepository()
        )

    async def load_profile(
        self,
        user_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> str | None:
        """返回非空画像文本；没有画像或读取失败返回 None。"""
        memory = await self.repository.get(user_id)
        if memory is None:
            return None
        text = _render_memory(memory.value)
        if not text:
            return None
        if token_budget is not None and token_budget > 0:
            text = truncate_to_tokens(text, token_budget)
        return text

    async def load_session_summary(
        self,
        conversation_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> str | None:
        """返回会话摘要文本；没有摘要或读取失败返回 None。"""
        summary = await self.summary_repository.get(conversation_id)
        if summary is None:
            return None
        text = _render_session_summary(summary.summary_json)
        if not text:
            return None
        if token_budget is not None and token_budget > 0:
            text = truncate_to_tokens(text, token_budget)
        return text

    async def compression_count(self, conversation_id: UUID) -> int:
        """返回会话摘要成功压缩次数；无摘要时按 0 处理。"""
        summary = await self.summary_repository.get(conversation_id)
        return int(summary.compression_count) if summary is not None else 0


def _render_memory(value: MemoryValue) -> str:
    """把固定 7 段画像渲染成带标签的纯文本。"""
    sections = (
        ("工作/学习", value.user.work_context.summary),
        ("个人背景", value.user.personal_context.summary),
        ("当前关注", value.user.top_of_mind.summary),
        ("认知偏好", value.user.cognitive_style.summary),
        ("近几个月", value.history.recent_months.summary),
        ("更早背景", value.history.earlier_context.summary),
        ("长期背景", value.history.long_term_background.summary),
    )
    lines = [f"- {label}: {summary.strip()}" for label, summary in sections if summary.strip()]
    return "\n".join(lines)


def _render_session_summary(summary: SessionSummary) -> str:
    """把 topics/facts/constraints 渲染成紧凑的会话摘要。

    topic 带上轮次 id，让模型在需要原始问答时调用 ``recall_round``；
    不展开中间消息。
    """
    lines: list[str] = []
    for topic in summary.topics:
        if not topic.summary.strip():
            continue
        rounds = ", ".join(str(item) for item in topic.source_round_ids)
        suffix = f" [rounds: {rounds}]" if rounds else ""
        lines.append(f"- 主题 {topic.name}: {topic.summary.strip()}{suffix}")
    for fact in summary.facts:
        if fact.content.strip():
            lines.append(f"- 事实: {fact.content.strip()}")
    for constraint in summary.constraints:
        if constraint.content.strip():
            lines.append(f"- 约束: {constraint.content.strip()}")
    return "\n".join(lines)
