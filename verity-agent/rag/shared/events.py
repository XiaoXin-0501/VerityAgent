"""RAG 侧的可观测性端口。

RAG 不依赖 Agent/Hooks 包，只向可选的 ``RagEventSink`` 发结构化事件；
Agent 层负责把事件适配到 HookDispatcher（见 ``agent/adapters/rag_events.py``）。
这样 RAG 内部不需要拼接日志摘要，也不会形成 ``rag -> agent`` 的反向依赖。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

__all__ = [
    "RagEventName",
    "RagEvent",
    "RagEventSink",
    "emit_rag_event",
]

logger = logging.getLogger(__name__)


class RagEventName(StrEnum):
    """RAG 事件名；值与 Agent ``HookEvent`` 保持一致，便于适配器直接映射。"""

    INDEX_START = "rag.index.start"
    INDEX_STAGE = "rag.index.stage"
    INDEX_WARNING = "rag.index.warning"
    INDEX_END = "rag.index.end"
    RETRIEVAL_START = "rag.retrieval.start"
    RETRIEVAL_END = "rag.retrieval.end"
    RERANK_END = "rag.rerank.end"
    GRAPH_END = "rag.graph.end"
    GRAPH_EXTRACTION_END = "rag.graph.extraction.end"
    CACHE_HIT = "rag.cache.hit"


@dataclass(slots=True, frozen=True)
class RagEvent:
    """一次 RAG 可观测事件；payload 只放标量/列表，由上层决定如何展示。"""

    name: RagEventName
    payload: dict[str, Any] = field(default_factory=dict)
    # 检索链路 id；入库任务没有请求 trace 时为空
    trace_id: str | None = None
    # 耗时（毫秒）；也可放在 payload 中供日志格式化
    duration_ms: int | None = None
    # 关联资源；适配器写入 HookContext.metadata
    doc_id: str | None = None
    kb_id: str | None = None


class RagEventSink(Protocol):
    """RAG 事件接收端；实现方必须是 fail-open（异常不能影响 RAG 主流程）。"""

    async def emit(self, event: RagEvent) -> None:
        """接收一个结构化事件。"""
        ...


async def emit_rag_event(sink: RagEventSink | None, event: RagEvent) -> None:
    """安全发送事件；未注入 sink 或 sink 异常时静默降级。"""
    if sink is None:
        return
    try:
        await sink.emit(event)
    except Exception:  # noqa: BLE001 - 观测失败不能影响检索/入库
        logger.debug("RAG 事件发送失败: %s", event.name.value, exc_info=True)
