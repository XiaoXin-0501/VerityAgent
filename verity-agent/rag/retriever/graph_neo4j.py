"""Neo4j 图扩展适配器：seed 实体 → 1 跳强关系 → next-hop 实体。

图只返回“下一跳是谁”，不直接返回 chunk；下一跳仍由 Weaviate 召回，避免
图证据和 BM25/向量抢召回职责。图不可用时返回空列表并记录 ``last_error``，
由 pipeline 转成 GRAPH_FAILED/GRAPH_LOW_EVIDENCE 告警，不中断整次检索。
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from typing import Any

from rag.shared.events import (
    RagEvent,
    RagEventName,
    RagEventSink,
    emit_rag_event,
)
from rag.shared.models import GraphExpansionPolicy
from store.neo4j import GraphStore

__all__ = ["Neo4jGraphExpander"]

logger = logging.getLogger(__name__)


class Neo4jGraphExpander:
    """基于 ``GraphStore.expand_next_hops`` 的检索扩展实现。"""

    def __init__(
        self,
        *,
        graph_store: GraphStore | None = None,
        event_sink: RagEventSink | None = None,
    ) -> None:
        """保存图存储门面与事件接收端；连接在首次查询时惰性建立。"""
        self.store = graph_store or GraphStore()
        self.event_sink = event_sink
        # 最近一次扩展失败原因；用 ContextVar 隔离并发请求，避免 A 请求的
        # 失败被 B 请求读到。pipeline 读取后写 GRAPH_FAILED 告警。
        self._last_error: ContextVar[str | None] = ContextVar(
            "rag_graph_last_error", default=None
        )

    @property
    def last_error(self) -> str | None:
        """当前异步上下文最近一次图扩展失败原因。"""
        return self._last_error.get()

    async def expand(
        self,
        *,
        seed_entities: list[str],
        kb_ids: list[Any],
        policy: GraphExpansionPolicy,
        trace_id: str | None = None,
    ) -> list[str]:
        """返回去重后的下一跳实体；按证据数/种子顺序稳定去重。"""
        self._last_error.set(None)
        if not seed_entities or not kb_ids or not policy.enabled:
            return []
        # v1 只允许 1 跳；max_hops 保留为配置/未来扩展，不在这里静默走多跳。
        if policy.max_hops > 1:
            logger.warning(
                "graph_max_hops=%s，v1 只执行 1 跳",
                policy.max_hops,
            )
        started = time.perf_counter()
        ids = list(dict.fromkeys(str(item) for item in kb_ids))
        next_hops: list[str] = []
        seen: set[str] = set()
        seed_kbs: set[str] = set()
        next_hop_kbs: set[str] = set()
        try:
            for seed in seed_entities:
                # 一次查询覆盖全部授权 KB：seed/关系/下一跳都允许来自
                # kb_ids 集合内的任意库，不再按单库串行展开。
                rows = await self.store.expand_next_hops(
                    seed=seed,
                    kb_ids=ids,
                    allowed_relations=policy.allowed_relations,
                    min_evidence=policy.min_evidence,
                    limit=max(1, policy.max_next_hops),
                    hub_as_intermediate=policy.hub_as_intermediate,
                )
                for row in rows:
                    name = str(row.get("next_hop_name") or "").strip()
                    if row.get("seed_kb_id"):
                        seed_kbs.add(str(row["seed_kb_id"]))
                    if row.get("next_hop_kb_id"):
                        next_hop_kbs.add(str(row["next_hop_kb_id"]))
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    next_hops.append(name)
                    if len(next_hops) >= policy.max_next_hops:
                        break
                if len(next_hops) >= policy.max_next_hops:
                    break
        except Exception as exc:  # noqa: BLE001 - 图失败必须降级，不抛穿
            self._last_error.set(str(exc))
            logger.warning("Neo4j 图扩展失败: %s", exc)
            await emit_rag_event(
                self.event_sink,
                RagEvent(
                    name=RagEventName.GRAPH_END,
                    payload={"status": "failed", "cause": str(exc)[:300]},
                    trace_id=trace_id,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                ),
            )
            return []
        duration_ms = int((time.perf_counter() - started) * 1000)
        await emit_rag_event(
            self.event_sink,
            RagEvent(
                name=RagEventName.GRAPH_END,
                payload={
                    "seed_count": len(seed_entities),
                    "kb_count": len(ids),
                    "seed_hit_kb_count": len(seed_kbs),
                    "next_hop_count": len(next_hops),
                    "next_hop_kb_count": len(next_hop_kbs),
                    "duration_ms": duration_ms,
                },
                trace_id=trace_id,
                duration_ms=duration_ms,
            ),
        )
        return next_hops
