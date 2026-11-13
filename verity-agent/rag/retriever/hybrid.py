"""混合召回接口（BM25 + vector）。

v1 由 retrieval pipeline 注入具体 Weaviate/Embedding 实现；本模块只定义
统一的候选结果形状，避免 fusion/rerank 直接依赖 store。
"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RetrievedChunk

__all__ = ["HybridRecallResult", "HybridRecaller"]


class HybridRecallResult:
    """一次混合召回的候选：BM25、向量与降级告警分开保留。"""

    def __init__(
        self,
        *,
        bm25: list[RetrievedChunk] | None = None,
        vector: list[RetrievedChunk] | None = None,
        warnings: list[str] | None = None,
    ) -> None:
        """保存两路候选与告警；空值统一成空列表。"""
        self.bm25 = bm25 or []
        self.vector = vector or []
        self.warnings = warnings or []


class HybridRecaller(Protocol):
    """混合召回协议；具体 Weaviate 实现由检索流水线注入。"""

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
        """返回 BM25 与向量两路候选；权限过滤必须在召回前完成。"""
        ...
