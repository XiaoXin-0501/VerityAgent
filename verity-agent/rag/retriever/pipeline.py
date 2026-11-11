"""RAG 检索流水线协议。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RetrievalRequest, RetrievalResult

__all__ = ["RetrievalPipeline"]


class RetrievalPipeline(Protocol):
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        ...

    async def close(self) -> None:
        ...
