"""邻居扩展接口。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import NeighborPolicy, RetrievedChunk

__all__ = ["NeighborExpander"]


class NeighborExpander(Protocol):
    async def expand(
        self,
        chunks: list[RetrievedChunk],
        policy: NeighborPolicy,
    ) -> list[RetrievedChunk]:
        """按 doc_id + chunk_index 补前后邻居，不改变主 chunk 的 citation。"""
        ...
