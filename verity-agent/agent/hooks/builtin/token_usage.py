"""内置 TokenUsageHook：汇总 token/成本、Redis 实时计数、回合结束写会话累计。"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from agent.llm import LLMUsage, merge_usage, ModelPricing
from config.hooks import TokenUsageHookConfig

from utils.time import utcnow

from ..counters import UsageCounterStore
from ..enum import HookEvent, UsageKind
from ..models import (
    CostRecord,
    HookContext,
    HookSpec,
    UsageAccumulator,
    UsageRecord,
    UsageSnapshot,
    calculate_cost,
)

logger = logging.getLogger(__name__)

__all__ = ["TokenUsageHook"]

_EVENTS = frozenset(
    {
        HookEvent.LLM_USAGE,
        HookEvent.EMBEDDING_USAGE,
        HookEvent.RERANK_USAGE,
        HookEvent.TURN_END,
        HookEvent.TURN_ERROR,
        HookEvent.RUN_END,
    }
)


def _add_cost(left: CostRecord | None, right: CostRecord | None) -> CostRecord | None:
    """累加两次成本记录；None 视为 0。"""
    if right is None:
        return left
    if left is None:
        return right
    return CostRecord(
        input_cost=left.input_cost + right.input_cost,
        output_cost=left.output_cost + right.output_cost,
        cached_cost=left.cached_cost + right.cached_cost,
        total_cost=left.total_cost + right.total_cost,
        currency=right.currency,
        pricing_version=right.pricing_version,
    )


class TokenUsageHook:
    def __init__(
        self,
        config: TokenUsageHookConfig,
        *,
        pricing: Mapping[str, ModelPricing] | None = None,
        counters: UsageCounterStore | None = None,
        conversation_repository: Any | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        """保存定价/计数器/仓储依赖，并合并配置中的价格表。"""
        self.config = config
        self.pricing = dict(pricing or {})
        self.counters = counters
        self.conversation_repository = conversation_repository
        self.logger = logger_ or logger
        self._accumulators: dict[str, UsageAccumulator] = {}
        self._chat_usage: dict[str, LLMUsage] = {}
        self._flushed: set[str] = set()

        prices: dict[str, ModelPricing] = {}
        for key, value in config.pricing.items():
            if isinstance(value, ModelPricing):
                prices[key] = value
            elif isinstance(value, Mapping):
                try:
                    prices[key] = ModelPricing.model_validate(
                        {"provider": "", "model": key, **dict(value)}
                    )
                except Exception:
                    logger.warning("TokenUsageHook 价格配置不合法: %s", key)
        self.pricing.update(prices)

        self.spec = HookSpec(
            name="builtin.token_usage",
            events=_EVENTS,
            priority=10,
            enabled=config.enabled,
        )

    async def __call__(self, context: HookContext) -> None:
        """按事件类型分派用量累加或回合/运行结束刷新。"""
        if context.event in (
            HookEvent.LLM_USAGE,
            HookEvent.EMBEDDING_USAGE,
            HookEvent.RERANK_USAGE,
        ):
            await self._on_usage(context)
        elif context.event in (HookEvent.TURN_END, HookEvent.TURN_ERROR):
            await self._flush_key(context, f"{context.trace_id}:{context.turn_id or ''}")
        elif context.event is HookEvent.RUN_END:
            prefix = f"{context.trace_id}:"
            for key in [item for item in self._accumulators if item.startswith(prefix)]:
                await self._flush_key(context, key)

    async def _on_usage(self, context: HookContext) -> None:
        """累加单次用量、计算成本，并按配置写 Redis 计数。"""
        payload = context.payload
        usage = getattr(payload, "usage", None)
        if usage is None:
            return
        if not isinstance(usage, LLMUsage):
            try:
                usage = LLMUsage.model_validate(usage)
            except Exception:
                logger.warning("TokenUsageHook usage 不合法: %s", type(usage).__name__)
                return
        call_id = context.call_id or getattr(payload, "call_id", None)
        if not call_id:
            call_id = f"{context.trace_id}:{context.event_id}"
        key = f"{context.trace_id}:{context.turn_id or ''}"
        accumulator = self._accumulators.setdefault(
            key, UsageAccumulator(key=key)
        )
        if call_id in accumulator.call_ids:
            return
        if self.counters is not None and self.config.redis_counters:
            if not await self.counters.seen(call_id):
                accumulator.call_ids.add(call_id)
                return

        kind = getattr(payload, "kind", None)
        if not isinstance(kind, UsageKind):
            if context.event is HookEvent.EMBEDDING_USAGE:
                kind = UsageKind.EMBEDDING
            elif context.event is HookEvent.RERANK_USAGE:
                kind = UsageKind.RERANK
            else:
                kind = UsageKind.CHAT
        model_id = getattr(payload, "model_id", None) or getattr(
            payload, "model", None
        )
        cost = calculate_cost(
            usage, self.pricing.get(str(model_id)) if model_id else None
        )
        record = UsageRecord(
            trace_id=context.trace_id,
            conversation_id=context.conversation_id,
            turn_id=context.turn_id,
            call_id=call_id,
            kind=kind,
            purpose=getattr(payload, "purpose", None),
            provider=getattr(payload, "provider", None),
            model=getattr(payload, "model", None),
            usage=usage,
            cost=cost,
        )
        accumulator.call_ids.add(call_id)
        accumulator.usage = merge_usage(accumulator.usage, usage)
        accumulator.cost = _add_cost(accumulator.cost, cost)
        accumulator.updated_at = utcnow()
        if kind is UsageKind.CHAT:
            self._chat_usage[key] = merge_usage(
                self._chat_usage.get(key, LLMUsage()), usage
            )
        if self.counters is not None and self.config.redis_counters:
            await self.counters.record(record, user_id=context.user_id)
        self.logger.info(
            "llm_usage kind=%s provider=%s model=%s call_id=%s input=%s output=%s cost=%s",
            kind.value,
            record.provider,
            record.model,
            call_id,
            usage.input_tokens,
            usage.output_tokens,
            record.cost.total_cost if record.cost else None,
        )

    async def _flush_key(self, context: HookContext, key: str) -> None:
        """把累计用量落盘为快照并清理内存状态。"""
        if key in self._flushed:
            return
        self._flushed.add(key)
        accumulator = self._accumulators.pop(key, None)
        chat_usage = self._chat_usage.pop(key, None)
        if accumulator is None:
            return
        snapshot = UsageSnapshot(
            conversation_id=context.conversation_id,
            turn_id=context.turn_id,
            usage=chat_usage or accumulator.usage,
            cost=accumulator.cost,
            call_count=len(accumulator.call_ids),
        )
        self.logger.info(
            "llm_usage_flush conversation_id=%s turn_id=%s input=%s output=%s calls=%s",
            context.conversation_id,
            context.turn_id,
            snapshot.usage.input_tokens,
            snapshot.usage.output_tokens,
            snapshot.call_count,
        )
        if (
            self.config.update_conversation_totals
            and self.conversation_repository is not None
            and context.conversation_id is not None
            and chat_usage is not None
            and (chat_usage.input_tokens or chat_usage.output_tokens)
        ):
            try:
                await self.conversation_repository.record_turn(
                    context.conversation_id,
                    input_tokens=chat_usage.input_tokens,
                    output_tokens=chat_usage.output_tokens,
                )
            except Exception as exc:
                # 投影更新失败不影响主流程；generation.usage 仍是权威。
                self.logger.warning("conversation token 投影更新失败: %s", exc)
