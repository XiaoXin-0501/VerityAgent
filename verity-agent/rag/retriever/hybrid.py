"""混合召回接口（BM25 + vector）。

v1 由 retrieval pipeline 注入具体 Weaviate/Embedding 实现；本模块只定义
统一的候选结果形状，避免 fusion/rerank 直接依赖 store。
"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RetrievedChunk

__all__ = ["HybridRecallResult", "HybridRecaller"]


class HybridRecallResult:
    def __init__(
        self,
        *,
        bm25: list[RetrievedChunk] | None = None,
        vector: list[RetrievedChunk] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        self.bm25 = bm25 or []
        self.vector = vector or []
        self.warnings = warnings or []


class HybridRecaller(Protocol):
    async def recall(
        self,
        query: str,
        *,
        kb_ids: list,
        permissions: frozenset[str],
        top_k_bm25: int,
        top_k_vector: int,
        filters=None,
    ) -> HybridRecallResult:
        ...
