"""RAG 入库流水线协议与依赖。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.models import IndexDocumentRequest, IndexingJobView

__all__ = ["IndexingPipeline"]


class IndexingPipeline(Protocol):
    """Indexer 门面背后的异步入库流水线。"""

    async def submit(self, request: IndexDocumentRequest) -> IndexingJobView:
        """提交入库任务；实现可选择后台执行。"""
        ...

    async def execute(self, request: IndexDocumentRequest) -> IndexingJobView:
        """同步执行入库；供 PG worker 复用同一编排。"""
        ...

    async def get_job(self, job_id: str) -> IndexingJobView | None:
        """查询任务；不存在返回 None。"""
        ...

    async def cancel(self, job_id: str) -> None:
        """幂等取消任务。"""
        ...

    async def delete(self, doc_id: UUID) -> IndexingJobView:
        """创建并执行删除任务。"""
        ...

    async def close(self) -> None:
        """停止后台任务并释放资源。"""
        ...
