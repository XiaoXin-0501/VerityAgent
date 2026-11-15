"""内置 TracingHook：v1 只维护 span 并写日志（exporter=none/log）。"""

from __future__ import annotations

import logging
from uuid import UUID

from config.hooks import TracingHookConfig

from ..enum import HookEvent, SpanStatus
from ..models import HookContext, HookSpec, SpanRecord

logger = logging.getLogger(__name__)

__all__ = ["TracingHook"]

_START_EVENTS = {
    HookEvent.RUN_START: "run",
    HookEvent.TURN_START: "turn",
    HookEvent.LLM_REQUEST_START: "llm.request",
    HookEvent.TOOL_EXECUTION_START: "tool.execute",
    HookEvent.TOOL_CALL_START: "tool.execute",
    HookEvent.RAG_RETRIEVAL_START: "rag.retrieve",
    HookEvent.SUMMARY_COMPRESS_START: "memory.compress",
    HookEvent.FILE_UPLOADED: "file.parse",
}
_END_EVENTS = {
    HookEvent.RUN_END: "run",
    HookEvent.RUN_ERROR: "run",
    HookEvent.TURN_END: "turn",
    HookEvent.TURN_ERROR: "turn",
    HookEvent.LLM_REQUEST_END: "llm.request",
    HookEvent.LLM_ERROR: "llm.request",
    HookEvent.TOOL_EXECUTION_END: "tool.execute",
    HookEvent.TOOL_RESULT: "tool.execute",
    HookEvent.TOOL_ERROR: "tool.execute",
    HookEvent.RAG_RETRIEVAL_END: "rag.retrieve",
    HookEvent.SUMMARY_COMPRESS_END: "memory.compress",
    HookEvent.FILE_PARSED: "file.parse",
}


class TracingHook:
    def __init__(self, config: TracingHookConfig) -> None:
        """保存 tracing 配置并构建起止事件的 HookSpec。"""
        self.config = config
        self._spans: dict[str, SpanRecord] = {}
        self.spec = HookSpec(
            name="builtin.tracing",
            events=frozenset(_START_EVENTS) | frozenset(_END_EVENTS),
            priority=700,
            sampling_rate=config.sample_rate,
            enabled=config.enabled,
        )

    async def __call__(self, context: HookContext) -> None:
        """在开始事件记录 span，在结束事件补齐状态并导出。"""
        name = _START_EVENTS.get(context.event) or _END_EVENTS.get(context.event)
        if name is None:
            return
        key = self._key(context, name)
        if context.event in _START_EVENTS:
            span = SpanRecord(
                trace_id=context.trace_id,
                span_id=context.span_id or context.event_id.hex,
                parent_span_id=context.parent_span_id,
                name=name,
                attributes={
                    "source": context.source.value,
                    "run_id": context.run_id or "",
                    "turn_id": context.turn_id or "",
                },
            )
            self._spans[key] = span
            self._export(span)
            return
        span = self._spans.pop(key, None)
        if span is None:
            return
        failed = context.event in (
            HookEvent.RUN_ERROR,
            HookEvent.TURN_ERROR,
            HookEvent.LLM_ERROR,
            HookEvent.TOOL_ERROR,
        )
        if context.event is HookEvent.TOOL_RESULT:
            failed = failed or str(
                getattr(context.payload, "status", "")
            ).upper() in {"FAILED", "CANCELLED"}
        span = span.model_copy(
            update={
                "ended_at": context.timestamp,
                "status": SpanStatus.ERROR if failed else SpanStatus.OK,
            }
        )
        self._export(span)

    def _key(self, context: HookContext, name: str) -> str:
        """用 trace/turn/call/name 生成 span 关联 key。"""
        return ":".join(
            [
                context.trace_id,
                context.turn_id or "",
                context.call_id or "",
                name,
            ]
        )

    def _export(self, span: SpanRecord) -> None:
        """按配置导出 span；none 表示不输出。"""
        if self.config.exporter == "none":
            return
        logger.info(
            "span name=%s trace_id=%s span_id=%s status=%s",
            span.name,
            span.trace_id,
            span.span_id,
            span.status.value,
        )
