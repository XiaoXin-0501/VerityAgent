"""HookDispatcher -> Agent HookPort 适配器。

Agent runtime 只依赖窄 ``HookPort.emit(event, payload, **kwargs)``；本适配器
把字符串事件转成 ``HookEvent``，并用 ``HookScope`` 补齐 run/turn/用户上下文。
Hook 失败不能影响主循环，调用方负责隔离异常。
"""

from __future__ import annotations

from typing import Any

from agent.hooks.dispatcher import HookDispatcher
from agent.hooks.enum import HookEvent, HookSource
from agent.hooks.scope import HookScope

__all__ = ["DispatcherHookAdapter"]


class DispatcherHookAdapter:
    """把 HookDispatcher 包装成运行时 HookPort。"""

    def __init__(
        self,
        dispatcher: HookDispatcher,
        *,
        source: HookSource = HookSource.AGENT,
    ) -> None:
        """保存 dispatcher 与默认事件来源。"""
        self.dispatcher = dispatcher
        self.source = source

    async def emit(
        self,
        event: str,
        payload: Any = None,
        **kwargs: Any,
    ) -> Any:
        """发送一条事件；trace/run/turn/user 等上下文从 kwargs 读取。"""
        trace_id = kwargs.pop("trace_id", None)
        run_id = kwargs.pop("run_id", None)
        turn_id = kwargs.pop("turn_id", None)
        user_id = kwargs.pop("user_id", None)
        conversation_id = kwargs.pop("conversation_id", None)
        source = kwargs.pop("source", self.source)
        scope = HookScope(
            self.dispatcher,
            trace_id=trace_id,
            run_id=run_id,
            user_id=user_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            source=source,
        )
        return await scope.emit(
            HookEvent(event),
            payload,
            call_id=kwargs.pop("call_id", None),
            span_id=kwargs.pop("span_id", None),
            parent_span_id=kwargs.pop("parent_span_id", None),
            metadata=kwargs.pop("metadata", None),
            duration_ms=kwargs.pop("duration_ms", None),
        )
