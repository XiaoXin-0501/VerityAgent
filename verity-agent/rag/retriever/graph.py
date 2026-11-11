"""图扩展接口：seed 实体 → 1 跳强关系 → next-hop 实体。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import GraphExpansionPolicy

__all__ = ["GraphExpander"]


class GraphExpander(Protocol):
    async def expand(
        self,
        *,
        seed_entities: list[str],
        kb_ids: list[str],
        policy: GraphExpansionPolicy,
    ) -> list[str]:
        """返回下一跳实体名；图不可用时返回空列表，不抛穿。"""
        ...
