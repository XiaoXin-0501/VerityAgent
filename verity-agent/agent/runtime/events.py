"""Agent 事件缓冲与事件序号。

v1 先提供内存 ring buffer；Redis Stream 适配器可在 EventPort 后面追加，
不改变 AgentLoop 的事件模型。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from agent.public import AgentEvent
from agent.runtime.enums import AgentEventType
from utils.time import utcnow

__all__ = ["AgentEventBuffer", "AgentEventPublisher"]

EventSink = Callable[[AgentEvent], Awaitable[None]]


class AgentEventBuffer:
    """单 run 的有界事件缓冲；支持 Last-Event-ID 回放。"""

    def __init__(self, *, max_events: int = 1000) -> None:
        """按上限保留最近事件，防止内存无界增长。"""
        self.max_events = max(1, int(max_events))
        self._events: deque[AgentEvent] = deque(maxlen=self.max_events)

    @property
    def last_event_id(self) -> str | None:
        """最后一条事件的 id；空缓冲返回 None。"""
        return self._events[-1].id if self._events else None

    def append(self, event: AgentEvent) -> None:
        """追加事件；超过 maxlen 自动淘汰最旧事件。"""
        self._events.append(event)

    def replay(self, after_event_id: str | None = None) -> list[AgentEvent]:
        """返回 ``after_event_id`` 之后的事件；None 返回全部。"""
        if not after_event_id:
            return list(self._events)
        after_seq = _seq_from_event_id(after_event_id)
        return [event for event in self._events if event.seq > after_seq]


class AgentEventPublisher:
    """生成单调 seq 的事件，并同时写入本地缓冲/订阅者。"""

    def __init__(
        self,
        *,
        run_id: str,
        turn_id: str,
        buffer: AgentEventBuffer,
        sink: EventSink | None = None,
    ) -> None:
        """保存 run/turn 与缓冲；sink 为空时只写缓冲。"""
        self.run_id = run_id
        self.turn_id = turn_id
        self.buffer = buffer
        self.sink = sink
        self._seq = 0

    async def emit(
        self,
        event_type: AgentEventType,
        data: dict[str, Any] | None = None,
        *,
        message_id: UUID | None = None,
        call_id: str | None = None,
    ) -> AgentEvent:
        """追加并广播一条事件；seq 在 run 内单调递增。"""
        self._seq += 1
        event = AgentEvent(
            seq=self._seq,
            event_type=event_type,
            run_id=self.run_id,
            turn_id=self.turn_id,
            message_id=message_id,
            call_id=call_id,
            data=data or {},
            created_at=utcnow(),
        )
        self.buffer.append(event)
        if self.sink is not None:
            await self.sink(event)
        return event


def _seq_from_event_id(event_id: str) -> int:
    """从 ``run_id:seq`` 取 seq；格式不合法按 1 处理，触发全量回放。"""
    try:
        return int(str(event_id).rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        return 0
