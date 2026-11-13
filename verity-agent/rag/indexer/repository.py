"""入库流水线需要的 PG 元数据接口。

IndexingPipeline 不直接依赖 ``DocumentRepository``，只依赖这个窄协议；
生产环境用 ``PgIndexingRepository`` 适配，测试可以注入内存 fake。
"""

from __future__ import annotations

from typing import Any, Protocol

from store.pg import DocumentRepository, KnowledgeBaseRepository

__all__ = ["IndexingRepository", "PgIndexingRepository"]


class IndexingRepository(Protocol):
    """入库任务的 PG 状态/统计写入协议。"""

    async def get_document(self, doc_id) -> Any | None:
        """读取文档元数据（含 index_version/status）。"""
        ...

    async def mark_processing(self, doc_id) -> None:
        """入库开始：pending/failed/completed → processing。"""
        ...

    async def acquire_index_lease(
        self,
        doc_id,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> None:
        """直接执行时获取文档租约；worker claim 路径可不实现。"""
        ...

    async def update_progress(
        self,
        doc_id,
        *,
        stage,
        progress: float,
        worker_id: str | None = None,
        include_deleted: bool = False,
    ) -> None:
        """写 index_stage/index_progress；删除流程需包含已软删行。"""
        ...

    async def mark_completed(
        self,
        doc_id,
        *,
        token_count: int,
        chunk_count: int,
        parse_manifest_key: str,
        index_version: int,
        worker_id: str | None = None,
    ) -> None:
        """提交 completed + 统计 + 版本 + 清租约。"""
        ...

    async def mark_failed(
        self,
        doc_id,
        *,
        error: str,
        worker_id: str | None = None,
        retry_backoff_s: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        """写 failed + 错误 + 退避。"""
        ...

    async def mark_deleting(self, doc_id) -> None:
        """软删除开始（status=deleting）。"""
        ...

    async def mark_deleted(self, doc_id) -> None:
        """跨存储删除完成（status=deleted）。"""
        ...

    async def set_description(self, doc_id, description: str | None) -> None:
        """写文档描述。"""
        ...

    async def bump_index_version(self, kb_id) -> int:
        """KB 索引版本 +1，返回新版本。"""
        ...


class PgIndexingRepository:
    """基于 ``DocumentRepository`` / ``KnowledgeBaseRepository`` 的适配器。"""

    def __init__(
        self,
        documents: DocumentRepository | None = None,
        knowledge_bases: KnowledgeBaseRepository | None = None,
    ) -> None:
        """保存两个仓库；不传时使用默认构造（共享全局连接池）。"""
        self.documents = documents or DocumentRepository()
        self.knowledge_bases = knowledge_bases or KnowledgeBaseRepository()

    async def get_document(self, doc_id) -> Any | None:
        """读取未删除文档。"""
        return await self.documents.get(doc_id)

    async def mark_processing(self, doc_id) -> None:
        """置 processing；状态机允许 pending/failed/completed → processing。"""
        await self.documents.mark_processing(doc_id)

    async def acquire_index_lease(
        self,
        doc_id,
        *,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> None:
        """直接执行入库时获取租约，补齐 index_locked_by/until。"""
        await self.documents.acquire_index_lease(
            doc_id,
            worker_id=worker_id,
            lease_seconds=lease_seconds,
        )

    async def update_progress(
        self,
        doc_id,
        *,
        stage,
        progress: float,
        worker_id: str | None = None,
        include_deleted: bool = False,
    ) -> None:
        """写入库进度；删除阶段需要更新已软删行。"""
        await self.documents.update_index_progress(
            doc_id,
            stage=stage,
            progress=progress,
            worker_id=worker_id,
            include_deleted=include_deleted,
        )

    async def mark_completed(
        self,
        doc_id,
        *,
        token_count: int,
        chunk_count: int,
        parse_manifest_key: str,
        index_version: int,
        worker_id: str | None = None,
    ) -> None:
        """提交完成状态。"""
        await self.documents.mark_index_completed(
            doc_id,
            token_count=token_count,
            chunk_count=chunk_count,
            parse_manifest_key=parse_manifest_key,
            index_version=index_version,
            worker_id=worker_id,
        )

    async def mark_failed(
        self,
        doc_id,
        *,
        error: str,
        worker_id: str | None = None,
        retry_backoff_s: float = 30.0,
        max_attempts: int = 5,
    ) -> None:
        """写失败状态与退避。"""
        await self.documents.mark_index_failed(
            doc_id,
            error=error,
            worker_id=worker_id,
            retry_backoff_s=retry_backoff_s,
            max_attempts=max_attempts,
        )

    async def mark_deleting(self, doc_id) -> None:
        """软删除开始。"""
        await self.documents.mark_deleting(doc_id)

    async def mark_deleted(self, doc_id) -> None:
        """删除完成。"""
        await self.documents.mark_deleted(doc_id)

    async def set_description(self, doc_id, description: str | None) -> None:
        """写文档描述。"""
        await self.documents.set_description(doc_id, description)

    async def bump_index_version(self, kb_id) -> int:
        """KB 版本 +1。"""
        return await self.knowledge_bases.bump_index_version(kb_id)
