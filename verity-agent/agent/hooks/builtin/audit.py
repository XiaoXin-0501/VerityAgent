"""内置 AuditHook：v1 写结构化审计日志，不落 PG。"""

from __future__ import annotations

import logging

from config.hooks import AuditHookConfig

from ..enum import AuditResult, HookEvent
from ..models import AuditEvent, HookContext, HookSpec
from ..redaction import RedactionPolicy

logger = logging.getLogger(__name__)

__all__ = ["AuditHook", "DEFAULT_EVENTS"]

DEFAULT_EVENTS = frozenset(
    {
        HookEvent.TOOL_CALL_START,
        HookEvent.TOOL_PERMISSION_DENIED,
        HookEvent.TOOL_RESULT,
        HookEvent.MESSAGE_APPENDED,
        HookEvent.MEMORY_UPDATE,
        HookEvent.PROMPT_AUDIT,
        HookEvent.RAG_RETRIEVAL_END,
        HookEvent.FILE_UPLOADED,
        HookEvent.FILE_PARSED,
        HookEvent.DELETION_START,
        HookEvent.DELETION_END,
        HookEvent.DELETION_FAILED,
    }
)


class AuditHook:
    def __init__(
        self,
        config: AuditHookConfig,
        redaction: RedactionPolicy,
        *,
        logger_: logging.Logger | None = None,
    ) -> None:
        """保存审计配置并构建 HookSpec（未配置事件时用默认集合）。"""
        self.config = config
        self.redaction = redaction
        self.logger = logger_ or logger
        events = (
            frozenset(HookEvent(item) for item in config.events)
            if config.events
            else DEFAULT_EVENTS
        )
        self.spec = HookSpec(
            name="builtin.audit",
            events=events,
            priority=500,
            enabled=config.enabled,
        )

    async def __call__(self, context: HookContext) -> None:
        """把事件转换为结构化审计记录并输出。"""
        payload = context.payload
        action = context.event.value
        result = AuditResult.SUCCESS
        if context.event in (
            HookEvent.TOOL_PERMISSION_DENIED,
            HookEvent.DELETION_FAILED,
            HookEvent.RUN_ERROR,
            HookEvent.TURN_ERROR,
        ):
            result = AuditResult.DENIED if "permission" in action else AuditResult.ERROR
        details = (
            self.redaction.redact_value(
                payload.model_dump(mode="json", by_alias=True)
            )
            if self.config.include_details
            else {}
        )
        event = AuditEvent(
            trace_id=context.trace_id,
            actor_id=context.user_id,
            action=action,
            resource_type=context.source.value,
            resource_id=(
                str(context.conversation_id) if context.conversation_id else None
            ),
            result=result,
            details=details,
        )
        self.logger.info(
            "audit action=%s resource_type=%s resource_id=%s result=%s trace_id=%s",
            event.action,
            event.resource_type,
            event.resource_id,
            event.result.value,
            event.trace_id,
        )
