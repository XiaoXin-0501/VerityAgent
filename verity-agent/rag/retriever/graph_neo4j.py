"""Neo4j 图扩展适配器：seed 实体 → 1 跳强关系 → next-hop 实体。

图只返回“下一跳是谁”，不直接返回 chunk；下一跳仍由 Weaviate 召回，避免
图证据和 BM25/向量抢召回职责。图不可用时返回空列表并记录 ``last_error``，
由 pipeline 转成 GRAPH_FAILED/GRAPH_LOW_EVIDENCE 告警，不中断整次检索。
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

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
    ) -> None:
        """保存图存储门面；连接在首次查询时惰性建立。"""
        self.store = graph_store or GraphStore()
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
        next_hops: list[str] = []
        seen: set[str] = set()
        try:
            for kb_id in kb_ids:
                for seed in seed_entities:
                    rows = await self.store.expand_next_hops(
                        seed=seed,
                        kb_id=str(kb_id),
                        allowed_relations=policy.allowed_relations,
                        min_evidence=policy.min_evidence,
                        limit=max(1, policy.max_next_hops),
                        hub_as_intermediate=policy.hub_as_intermediate,
                    )
                    for row in rows:
                        name = str(row.get("next_hop_name") or "").strip()
                        if not name or name in seen:
                            continue
                        seen.add(name)
                        next_hops.append(name)
                        if len(next_hops) >= policy.max_next_hops:
                            return next_hops
        except Exception as exc:  # noqa: BLE001 - 图失败必须降级，不抛穿
            self._last_error.set(str(exc))
            logger.warning("Neo4j 图扩展失败: %s", exc)
            return []
        return next_hops
