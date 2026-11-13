"""RAG 检索流水线协议。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RetrievalRequest, RetrievalResult

__all__ = ["RetrievalPipeline"]


class RetrievalPipeline(Protocol):
    """Retriever 门面背后的只读检索流水线。"""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """执行权限过滤、召回、融合、重排与引用组装。"""
        ...

    async def close(self) -> None:
        """释放缓存/连接资源。"""
        ...
