"""用户跨存储删除 Worker：PG 软删 -> Weaviate UserFact -> 完成/重试。"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from config.memory import MemoryConfig
from store.minio import MinioStorage
from store.pg.models.enums import SessionFileStatus
from store.pg.repositories import SessionFileRepository, UserRepository
from store.weaviate import WeaviateStore
from utils.time import utcnow

__all__ = ["UserDeletionWorker"]

logger = logging.getLogger(__name__)


class UserDeletionWorker:
    """轮询 ``users.status=deleting`` 并完成跨存储清理。"""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        repository: UserRepository | None = None,
        vector_store: WeaviateStore | None = None,
        files: SessionFileRepository | None = None,
        storage: MinioStorage | None = None,
        hook_dispatcher=None,
    ) -> None:
        """保存配置/仓储；不建立连接。"""
        self.config = config
        self.repository = repository or UserRepository()
        self.vector_store = vector_store or WeaviateStore()
        self.files = files or SessionFileRepository()
        self.storage = storage or MinioStorage()
        self.hook_dispatcher = hook_dispatcher
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """请求 worker 退出。"""
        self._stop.set()

    async def run_forever(self) -> None:
        """轮询待删除用户；没有任务时等待固定间隔。"""
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - worker 不因单轮异常退出
                logger.exception("用户删除 worker 批次异常")
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.config.deletion_scan_interval_s,
                )
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> bool:
        """扫描一批到期用户；返回是否处理了任务。"""
        # 在 SQL 层过滤重试时间，避免队首失败用户长期占用批次导致饥饿。
        users = await self.repository.list_deletion_ready(
            limit=self.config.deletion_batch_size,
        )
        for user in users:
            try:
                await self.vector_store.delete_user_facts(user.id)
                await self._delete_session_files(user.id)
                await self.repository.mark_deleted(user.id)
            except Exception as exc:  # noqa: BLE001 - 单用户失败写回退
                logger.warning("用户跨存储删除失败: %s", user.id, exc_info=True)
                await self.repository.record_delete_failure(
                    user.id,
                    str(exc)[:500],
                    retry_at=utcnow()
                    + timedelta(seconds=self.config.deletion_retry_backoff_s),
                )
        return bool(users)

    async def _delete_session_files(self, user_id) -> None:
        """删除用户上传的会话文件原件/解析产物；按对象前缀幂等。"""
        files = await self.files.list_by_uploader(user_id, limit=100)
        for item in files:
            if item.status is SessionFileStatus.DELETED:
                continue
            if item.status is not SessionFileStatus.DELETING:
                await self.files.mark_deleting(item.id)
            await self.storage.delete_session_file(
                item.conversation_id,
                item.id,
                extension=item.extension,
            )
            await self.files.mark_deleted(item.id)
