"""内置 LoggingHook：结构化日志。"""

from __future__ import annotations

import logging
from typing import Any

from config.hooks import LoggingHookConfig
from ..enum import HookEvent, HookExecutionMode
from ..models import HookContext, HookSpec
from ..redaction import RedactionPolicy

__all__ = ["LoggingHook", "DEFAULT_EVENTS"]

DEFAULT_EVENTS = frozenset(
    {
        HookEvent.RUN_START,
        HookEvent.RUN_END,
        HookEvent.RUN_ERROR,
        HookEvent.TURN_START,
        HookEvent.TURN_END,
        HookEvent.TURN_ERROR,
        HookEvent.LLM_ERROR,
        HookEvent.LLM_USAGE,
        HookEvent.TOOL_RESULT,
        HookEvent.TOOL_ERROR,
        HookEvent.MESSAGE_APPENDED,
        HookEvent.MESSAGE_FINALIZED,
        HookEvent.RAG_RETRIEVAL_END,
        HookEvent.FILE_PARSED,
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
                    context.payload.model_dump(mode="json", by_alias=True),
                    max_bytes=self.config.max_payload_bytes,
                )
                if self.config.include_payload
                else None
            ),
        }
        self.logger.log(
            self._level,
            "hook_event %s",
            context.event.value,
            extra={"hook_data": data},
        )
