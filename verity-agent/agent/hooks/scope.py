"""一次运行的 Hook 上下文门面。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from utils.ids import new_run_id, new_request_id, uuid7
from utils.time import utcnow

from .dispatcher import HookDispatcher
from .enum import HookEvent, HookSource
from .models import HookContext, HookDispatchResult, HookPayload

__all__ = ["HookScope"]


class HookScope:
    """补 trace/run/user/conversation，避免每个 emit 点重复传公共字段。"""

    def __init__(
        self,
        dispatcher: HookDispatcher,
        *,
        trace_id: str | None = None,
        run_id: str | None = None,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
        turn_id: str | None = None,
        source: HookSource = HookSource.AGENT,
    ) -> None:
        """保存公共上下文；缺省 trace/run id 自动生成。"""
        self.dispatcher = dispatcher
        self.trace_id = trace_id or new_request_id()
        self.run_id = run_id or new_run_id()
        self.user_id = user_id
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.source = source

    async def emit(
        self,
        event: HookEvent,
        payload: HookPayload | dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
        call_id: str | None = None,
        source: HookSource | None = None,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        duration_ms: int | None = None,
    ) -> HookDispatchResult:
        """补齐公共字段后分发事件；payload 可为模型或字典。"""
        if payload is None:
            payload_model = HookPayload()
        elif isinstance(payload, HookPayload):
            payload_model = payload
        else:
            payload_model = HookPayload.model_validate(payload)
        context = HookContext(
            event=event,
            event_id=uuid7(),
            timestamp=utcnow(),
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            source=source or self.source,
            user_id=self.user_id,
            conversation_id=self.conversation_id,
            turn_id=turn_id or self.turn_id,
            run_id=self.run_id,
            call_id=call_id,
            payload=payload_model,
            metadata=metadata or {},
            duration_ms=duration_ms,
        )
        return await self.dispatcher.emit(context)

    async def flush(self) -> None:
        """等待该 scope 触发的后台 Hook 完成。"""
        await self.dispatcher.flush()

    async def close(self) -> None:
        """刷新后结束 scope（不关闭共享 dispatcher）。"""
        await self.flush()
