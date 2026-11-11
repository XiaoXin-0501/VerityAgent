"""RAG 入库流水线协议与依赖。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.models import IndexDocumentRequest, IndexingJobView

__all__ = ["IndexingPipeline"]


class IndexingPipeline(Protocol):
    """Indexer 门面背后的异步入库流水线。"""

    async def submit(self, request: IndexDocumentRequest) -> IndexingJobView:
        ...

    async def get_job(self, job_id: str) -> IndexingJobView | None:
        ...

    async def cancel(self, job_id: str) -> None:
        ...

    async def delete(self, doc_id: UUID) -> IndexingJobView:
        ...

    async def close(self) -> None:
        ...
