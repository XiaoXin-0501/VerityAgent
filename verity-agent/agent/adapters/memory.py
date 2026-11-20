"""用户长期画像读取适配器。"""

from __future__ import annotations

from uuid import UUID

from store.pg.models.memory import MemoryValue
from store.pg.models.conversation import SessionSummary
from store.pg.repositories import (
    ConversationSummaryRepository,
    UserMemoryRepository,
)
from utils.time import ensure_utc
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
        text, _ = await self.load_session_summary_state(
            conversation_id,
            token_budget=token_budget,
        )
        return text

    async def load_session_summary_state(
        self,
        conversation_id: UUID,
        *,
        token_budget: int | None = None,
    ) -> tuple[str | None, int]:
        """返回 (会话摘要文本, covered_through_seq)。

        ``covered_through_seq`` 用于历史窗口加载：只加载摘要尚未覆盖的消息，
        避免摘要与工作窗口重复进 Prompt。
        """
        summary = await self.summary_repository.get(conversation_id)
        if summary is None:
            return None, 0
        text = _render_session_summary(summary.summary_json)
        if text and token_budget is not None and token_budget > 0:
            text = truncate_to_tokens(text, token_budget)
        return text or None, int(summary.covered_through_seq)

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
    """把 topics/facts/constraints 渲染成紧凑但保留判断依据的会话摘要。

    - topic 带讨论次数、最近讨论时间与轮次 id，供模型判断话题热度并调用
      ``recall_round``；
    - fact/constraint 带置信度、确认次数、最近确认时间与所属主题，供模型
      判断信息稳定性和时效；
    - 不展开中间消息，避免把摘要变成完整历史。
    """
    topic_names = {topic.id: topic.name for topic in summary.topics}
    lines: list[str] = []
    for topic in summary.topics:
        if not topic.summary.strip():
            continue
        rounds = ", ".join(str(item) for item in topic.source_round_ids)
        suffix = f" [rounds: {rounds}]" if rounds else ""
        lines.append(
            f"- 主题 {topic.name}（讨论 {topic.mention_count} 次，"
            f"最近 {_format_time(topic.last_mentioned_at)}）: "
            f"{topic.summary.strip()}{suffix}"
        )
    for fact in summary.facts:
        content = fact.content.strip()
        if not content:
            continue
        topic_name = topic_names.get(fact.topic_id)
        meta = (
            f"置信度 {fact.confidence:.2f}，确认 {fact.confirmation_count} 次，"
            f"最近 {_format_time(fact.last_confirmed_at)}"
        )
        if topic_name:
            meta = f"主题 {topic_name}；{meta}"
        lines.append(f"- 事实: {content}（{meta}）")
    for constraint in summary.constraints:
        content = constraint.content.strip()
        if not content:
            continue
        meta = (
            f"置信度 {constraint.confidence:.2f}，"
            f"确认 {constraint.confirmation_count} 次，"
            f"最近 {_format_time(constraint.last_confirmed_at)}"
        )
        lines.append(f"- 约束: {content}（{meta}）")
    return "\n".join(lines)


def _format_time(value) -> str:
    """把摘要时间格式化为 UTC 分钟精度；缺失/异常时返回 ``未知``。"""
    if value is None:
        return "未知"
    try:
        return ensure_utc(value).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(value)
