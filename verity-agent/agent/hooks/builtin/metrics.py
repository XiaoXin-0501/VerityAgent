"""内置 MetricsHook：v1 写日志 sink，指标名/标签保持低基数。"""

from __future__ import annotations

import logging
from typing import Any

from config.hooks import MetricsHookConfig

from ..enum import HookEvent, MetricKind
from ..models import HookContext, HookSpec, MetricRecord

logger = logging.getLogger(__name__)

__all__ = ["MetricsHook"]

_EVENTS = frozenset(
    {
        HookEvent.LLM_USAGE,
        HookEvent.LLM_REQUEST_END,
        HookEvent.TOOL_RESULT,
        HookEvent.TURN_END,
        HookEvent.TURN_ERROR,
        HookEvent.RAG_RETRIEVAL_END,
        HookEvent.RAG_RERANK_END,
        HookEvent.SUMMARY_COMPRESS_END,
    }
)


class MetricsHook:
    def __init__(self, config: MetricsHookConfig) -> None:
        """保存指标配置并构建 HookSpec。"""
        self.config = config
        self.spec = HookSpec(
            name="builtin.metrics",
            events=_EVENTS,
            priority=800,
            sampling_rate=config.sample_rate,
            enabled=config.enabled,
        )

    async def __call__(self, context: HookContext) -> None:
        """把事件转换为指标记录并写入配置的 sink。"""
        if self.config.sink == "noop":
            return
        records = self._records(context)
        for record in records:
            logger.info(
                "metric name=%s kind=%s value=%s unit=%s tags=%s",
                record.name,
                record.kind.value,
                record.value,
                record.unit,
                record.tags,
            )

    def _records(self, context: HookContext) -> list[MetricRecord]:
        """按事件类型生成一条或多条指标记录。"""
        payload = context.payload
        tags: dict[str, str] = {"event": context.event.value}
        name = f"{self.config.prefix}_unknown"
        kind = MetricKind.COUNTER
        value = 1.0
        unit: str | None = "count"

        if context.event is HookEvent.LLM_USAGE:
            name = f"{self.config.prefix}_llm_tokens_total"
            usage = getattr(payload, "usage", None)
            value = float(getattr(usage, "total_tokens", 0) or 0)
            unit = "tokens"
            tags.update(
                provider=str(getattr(payload, "provider", "")),
                model=str(getattr(payload, "model", "")),
            )
            return [
                MetricRecord(
                    name=name, kind=kind, value=value, unit=unit, tags=tags
                )
            ]
        if context.event is HookEvent.TOOL_RESULT:
            name = f"{self.config.prefix}_tool_calls_total"
            tags.update(
                tool=str(getattr(payload, "tool_name", "")),
                status=str(getattr(payload, "status", "")),
            )
        elif context.event in (HookEvent.TURN_END, HookEvent.TURN_ERROR):
            name = f"{self.config.prefix}_turn_duration_ms"
            value = float(context.duration_ms or 0)
            unit = "ms"
            kind = MetricKind.HISTOGRAM
            tags["status"] = context.event.value
        elif context.event is HookEvent.RAG_RETRIEVAL_END:
            name = f"{self.config.prefix}_rag_retrieval_duration_ms"
            value = float(getattr(payload, "duration_ms", 0) or 0)
            unit = "ms"
            kind = MetricKind.HISTOGRAM
        elif context.event is HookEvent.LLM_REQUEST_END:
            name = f"{self.config.prefix}_llm_requests_total"
            tags["status"] = "completed"
        return [
            MetricRecord(name=name, kind=kind, value=value, unit=unit, tags=tags)
        ]
