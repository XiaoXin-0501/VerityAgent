"""内置 LoggingHook：结构化日志。"""

from __future__ import annotations

import logging
from typing import Any

from config.hooks import LoggingHookConfig
from ..enum import HookEvent, HookExecutionMode
from ..models import HookContext, HookSpec
from ..redaction import RedactionPolicy

__all__ = ["LoggingHook", "DEFAULT_EVENTS"]

# 默认只记录“分析系统/性能”真正有用的事件；run/turn 生命周期由 AgentLoop
# 的完成摘要负责，避免同一件事在 app.log 里出现两遍。
DEFAULT_EVENTS = frozenset(
    {
        HookEvent.LLM_ERROR,
        HookEvent.LLM_USAGE,
        HookEvent.TOOL_RESULT,
        HookEvent.TOOL_ERROR,
        HookEvent.FILE_PARSED,
        HookEvent.FILE_PARSE_FAILED,
        HookEvent.DELETION_FAILED,
    }
)


class LoggingHook:
    def __init__(
        self,
        config: LoggingHookConfig,
        redaction: RedactionPolicy,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        """保存日志配置与脱敏策略，构建 HookSpec。"""
        self.config = config
        self.redaction = redaction
        self.logger = logger or logging.getLogger("verity.hooks")
        events = (
            frozenset(HookEvent(item) for item in config.events)
            if config.events
            else DEFAULT_EVENTS
        )
        try:
            mode = HookExecutionMode(config.mode)
        except ValueError:
            mode = HookExecutionMode.BACKGROUND
        self.spec = HookSpec(
            name="builtin.logging",
            events=events,
            priority=900,
            mode=mode,
            timeout_s=1.0,
            enabled=config.enabled,
        )
        self._level = getattr(logging, config.level.upper(), logging.INFO)

    async def __call__(self, context: HookContext) -> None:
        """输出结构化日志；payload 按配置脱敏并截断。"""
        payload = context.payload.model_dump(mode="json", by_alias=True)
        data: dict[str, Any] = {
            "event": context.event.value,
            "trace_id": context.trace_id,
            "run_id": context.run_id,
            "conversation_id": str(context.conversation_id)
            if context.conversation_id
            else None,
            "turn_id": context.turn_id,
            "call_id": context.call_id,
            "source": context.source.value,
            "duration_ms": context.duration_ms,
            "event_id": str(context.event_id),
            "payload": (
                self.redaction.payload_summary(
                    payload,
                    max_bytes=self.config.max_payload_bytes,
                )
                if self.config.include_payload
                else None
            ),
        }
        self.logger.log(
            self._level,
            "%s",
            _format_event(context, payload),
            extra={"hook_data": data},
        )


def _format_event(context: HookContext, payload: dict[str, Any]) -> str:
    """把 Hook 事件转成一行人能读懂的中文摘要。"""
    event = context.event
    duration = context.duration_ms
    call = (
        f"call={context.call_id[:8]} | "
        if context.call_id
        else ""
    )
    if event is HookEvent.LLM_USAGE:
        usage = payload.get("usage") or {}
        latency = payload.get("latency_ms") or duration
        return (
            f"LLM 调用 | {call}model={payload.get('model_id') or payload.get('model') or '-'}"
            f" | purpose={payload.get('purpose') or '-'}"
            f" | 延迟={latency or 0}ms"
            f" | tokens 输入={usage.get('input_tokens', 0)}"
            f" 输出={usage.get('output_tokens', 0)}"
            f" 缓存={usage.get('cached_input_tokens', 0)}"
            f" | finish={payload.get('finish_reason') or '-'}"
        )
    if event is HookEvent.LLM_ERROR:
        return (
            f"LLM 失败 | {call}model={payload.get('model') or '-'}"
            f" | purpose={payload.get('purpose') or '-'}"
            f" | error={payload.get('error_code') or '-'}"
            f" | retryable={payload.get('retryable', False)}"
            f" | attempt={payload.get('attempt', 1)}"
        )
    if event in {HookEvent.TOOL_RESULT, HookEvent.TOOL_ERROR}:
        flags = []
        if payload.get("cache_hit"):
            flags.append("cache")
        if payload.get("idempotent_replay"):
            flags.append("replay")
        if payload.get("truncated"):
            flags.append("truncated")
        suffix = f" | flags={','.join(flags)}" if flags else ""
        return (
            f"工具{'失败' if event is HookEvent.TOOL_ERROR else '结果'}"
            f" | {call}tool={payload.get('tool_name') or '-'}"
            f" | status={payload.get('status') or '-'}"
            f" | 耗时={payload.get('duration_ms') or duration or 0}ms"
            f" | 输出={payload.get('content_bytes', 0)}B"
            f"{suffix}"
            + (
                f" | error={payload.get('error_code')}"
                if payload.get("error_code")
                else ""
            )
        )
    if event in {HookEvent.FILE_PARSED, HookEvent.FILE_PARSE_FAILED}:
        return (
            f"文件{'解析失败' if event is HookEvent.FILE_PARSE_FAILED else '解析'}"
            f" | file={payload.get('filename') or '-'}"
            f" | size={payload.get('size_bytes', 0)}B"
            f" | status={payload.get('status') or '-'}"
            f" | 耗时={payload.get('duration_ms') or duration or 0}ms"
        )
    if event is HookEvent.DELETION_FAILED:
        return (
            f"删除失败 | error={payload.get('error_code') or '-'}"
            f" | 耗时={duration or 0}ms"
        )
    if event is HookEvent.RUN_END:
        usage = payload.get("usage") or {}
        return (
            f"Run 结束 | status={payload.get('status') or '-'}"
            f" | 耗时={payload.get('duration_ms') or duration or 0}ms"
            f" | tokens 输入={usage.get('input_tokens', 0)}"
            f" 输出={usage.get('output_tokens', 0)}"
        )
    if event is HookEvent.TURN_END:
        usage = payload.get("usage") or {}
        return (
            f"回合结束 | status={payload.get('status') or '-'}"
            f" | 消息={payload.get('message_count', 0)}"
            f" | tokens 输入={usage.get('input_tokens', 0)}"
            f" 输出={usage.get('output_tokens', 0)}"
            f" | 耗时={duration or 0}ms"
        )
    # 兜底：只输出少量标量字段，避免把整段 payload 塞进日志。
    parts = [f"事件={event.value}"]
    if duration is not None:
        parts.append(f"耗时={duration}ms")
    for key, value in payload.items():
        if len(parts) >= 8:
            break
        if isinstance(value, (str, int, float, bool)) and value not in {"", None}:
            parts.append(f"{key}={value}")
    return " | ".join(parts)
