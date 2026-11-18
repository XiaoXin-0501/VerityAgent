"""RAG 结构化事件 -> Agent Hooks 适配器。

RAG 只依赖自己的 ``RagEventSink``，Agent 层在这里把事件映射成
``HookEvent`` 并交给 ``HookDispatcher``；依赖方向保持 ``agent -> rag``。
"""

from __future__ import annotations

import logging
from typing import Any

from agent.hooks.dispatcher import HookDispatcher
from agent.hooks.enum import HookEvent, HookSource
from agent.hooks.models import HookContext, HookPayload
from rag.shared.events import RagEvent
from utils.ids import new_request_id

__all__ = ["HookRagEventSink"]

logger = logging.getLogger(__name__)


class HookRagEventSink:
    """把 RAG 事件转成 Hook 事件；失败只记 debug，不影响 RAG。"""

    def __init__(self, dispatcher: HookDispatcher) -> None:
        """保存共享 HookDispatcher；生命周期由应用管理。"""
        self.dispatcher = dispatcher

    async def emit(self, event: RagEvent) -> None:
        """发送一个 RAG 事件；事件名不在 HookEvent 中时静默忽略。"""
        try:
            hook_event = HookEvent(event.name.value)
        except ValueError:
            logger.debug("未映射的 RAG 事件: %s", event.name.value)
            return
        payload = dict(event.payload)
        if event.duration_ms is not None:
            payload.setdefault("duration_ms", event.duration_ms)
        metadata: dict[str, Any] = {}
        if event.doc_id:
            metadata["doc_id"] = event.doc_id
        if event.kb_id:
            metadata["kb_id"] = event.kb_id
        try:
            context = HookContext(
                event=hook_event,
                trace_id=event.trace_id or new_request_id(),
                source=HookSource.RAG,
                payload=HookPayload.model_validate(payload),
                metadata=metadata,
                duration_ms=event.duration_ms,
            )
            await self.dispatcher.emit(context)
        except Exception:  # noqa: BLE001 - 观测失败不能影响检索/入库
            logger.debug("RAG Hook 事件发送失败: %s", event.name.value, exc_info=True)
