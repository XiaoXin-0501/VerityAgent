"""上下文预算规划：按 token 预算选择历史窗口，而不是固定条数。

工作窗口只加载摘要 ``covered_through_seq`` 之后的消息，避免与长期摘要重复；
随后按完整轮次从新到旧装入模型预算。当前用户回合始终保留。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

from agent.context.models import ContextSection
from agent.message.models import RawMessage, StoredMessage, UserMessage
from utils.serialization import json_dumps
from utils.tokens import estimate_tokens

__all__ = ["ContextBudgetPlan", "ContextBudgetPlanner", "message_tokens"]


@dataclass(frozen=True, slots=True)
class ContextBudgetPlan:
    """一次历史窗口预算规划结果。"""

    messages: list[StoredMessage]
    history_budget: int
    token_estimate: int
    dropped_messages: int
    kept_rounds: int
    truncated: bool


class ContextBudgetPlanner:
    """按模型窗口、输出预留和 section 预算选择工作窗口消息。"""

    def __init__(
        self,
        *,
        max_context_tokens: int = 8192,
        reserved_output_tokens: int = 2048,
        max_history_messages: int = 400,
    ) -> None:
        """保存模型窗口与安全上限；不读取消息。"""
        self.max_context_tokens = max(256, int(max_context_tokens))
        self.reserved_output_tokens = max(0, int(reserved_output_tokens))
        # 安全阀：防止异常会话一次加载过多；实际选择由 token 预算决定。
        self.max_history_messages = max(1, int(max_history_messages))

    def plan(
        self,
        messages: Sequence[StoredMessage],
        *,
        system_prompt: str | None = None,
        sections: Sequence[ContextSection] = (),
        tool_tokens: int = 0,
    ) -> ContextBudgetPlan:
        """从完整未摘要窗口中选择能放进预算的完整轮次。"""
        items = list(messages)[-self.max_history_messages :]
        available = self.max_context_tokens - self.reserved_output_tokens
        available -= estimate_tokens(system_prompt or "")
        available -= max(0, int(tool_tokens))
        for section in sections:
            available -= section.token_estimate or estimate_tokens(section.content)
        budget = max(64, available)

        # 安全上限截断或丢弃半轮，也算本次窗口发生裁剪。
        truncated = len(items) < len(messages)
        rounds = _group_rounds(items)
        # 安全上限按最近 N 条截断时，最老一组可能只有 assistant/tool 而缺少
        # 起始 user 消息；丢弃这条不完整轮次，避免把半条工具链装进 Prompt。
        if rounds and not any(
            isinstance(item.message, UserMessage) for item in rounds[0]
        ):
            rounds = rounds[1:]
            truncated = True
        if not rounds:
            return ContextBudgetPlan(
                messages=[],
                history_budget=budget,
                token_estimate=0,
                dropped_messages=len(messages),
                kept_rounds=0,
                truncated=bool(messages),
            )

        chosen: list[list[StoredMessage]] = []
        remaining = budget
        # 最后一条通常是当前用户问题，任何情况下都要保留。
        last_message = items[-1] if items else None
        for index, round_messages in enumerate(reversed(rounds)):
            round_tokens = sum(message_tokens(item) for item in round_messages)
            if round_tokens <= remaining:
                chosen.insert(0, round_messages)
                remaining -= round_tokens
                continue
            if index == 0:
                # 当前轮本身超预算：至少保留最后一条消息（通常是当前用户
                # 问题），再按预算向前补最近的同轮消息。
                partial = [last_message] if last_message is not None else []
                partial_budget = remaining - (
                    message_tokens(last_message) if last_message is not None else 0
                )
                for message in reversed(round_messages[:-1]):
                    tokens = message_tokens(message)
                    if tokens <= partial_budget:
                        partial.insert(0, message)
                        partial_budget -= tokens
                    else:
                        break
                if partial:
                    chosen.insert(0, partial)
                truncated = True
            else:
                truncated = True
            # 更早的轮次不再尝试：保持窗口连续，避免中间空洞。
            break

        selected = [item for group in chosen for item in group]
        return ContextBudgetPlan(
            messages=selected,
            history_budget=budget,
            token_estimate=sum(message_tokens(item) for item in selected),
            dropped_messages=max(0, len(messages) - len(selected)),
            kept_rounds=len(chosen),
            truncated=truncated,
        )


def message_tokens(stored: StoredMessage) -> int:
    """估算一条 StoredMessage 的 token；附件按 1 token 计。"""
    message = stored.message
    if isinstance(message, RawMessage):
        return 0
    total = estimate_tokens(message.content or "")
    for thinking in getattr(message, "thinking", []) or []:
        total += estimate_tokens(getattr(thinking, "text", "") or "")
    for call in getattr(message, "tool_calls", []) or []:
        total += estimate_tokens(getattr(call, "name", "") or "")
        total += estimate_tokens(
            json_dumps(getattr(call, "arguments", None) or {})
        )
    total += len(getattr(message, "images", []) or [])
    total += len(getattr(message, "files", []) or [])
    # ToolMessage.json 只落库/给前端，不回喂 LLM；预算只计实际发送的
    # content/thinking/tool_calls，避免结构化数据重复占预算。
    return total


def _group_rounds(messages: Sequence[StoredMessage]) -> list[list[StoredMessage]]:
    """按 round_id 分组；旧数据按“用户消息开始新轮次”兜底。"""
    groups: list[list[StoredMessage]] = []
    current: list[StoredMessage] = []
    current_key: UUID | None = None
    for stored in messages:
        message = stored.message
        round_id = stored.round_id
        is_user = isinstance(message, UserMessage)
        if current:
            if round_id is not None and round_id != current_key:
                groups.append(current)
                current = []
                current_key = None
            elif round_id is None and is_user:
                groups.append(current)
                current = []
                current_key = None
        current.append(stored)
        if round_id is not None:
            current_key = round_id
    if current:
        groups.append(current)
    return groups
