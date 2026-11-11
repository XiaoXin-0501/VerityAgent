"""Rerank 接口。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RerankPolicy, RetrievedChunk

__all__ = ["Reranker"]


class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        policy: RerankPolicy,
    ) -> list[RetrievedChunk]:
        """返回带 rerank_score 的 chunk；失败由 pipeline 降级为 warning。"""
        ...
