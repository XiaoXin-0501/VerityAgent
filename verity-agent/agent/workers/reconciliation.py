"""长期事实数量对账 Worker：Weaviate active count -> PG user_memory.fact_count。"""

from __future__ import annotations

import asyncio
import logging

from config.memory import MemoryConfig
from store.pg.models.memory import MemoryChange
from store.pg.repositories import UserMemoryRepository

__all__ = ["FactReconciliationWorker"]

logger = logging.getLogger(__name__)


class FactReconciliationWorker:
    """定期校正 PG fact_count 投影。"""

    def __init__(
        self,
        config: MemoryConfig,
        *,
        facts_store,
        memory_repository: UserMemoryRepository | None = None,
    ) -> None:
        """保存配置/事实存储/画像仓储。"""
        self.config = config
        self.facts_store = facts_store
        self.memories = memory_repository or UserMemoryRepository()
        self._offset = 0
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """请求 worker 退出。"""
        self._stop.set()

    async def run_forever(self) -> None:
        """周期执行对账；停止信号优先。"""
        while not self._stop.is_set():
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 单轮失败继续下一轮
                logger.exception("事实对账 worker 异常")
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.config.reconciliation_interval_s,
                )
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> int:
        """对账一批用户；返回校正数量。"""
        user_ids = await self.memories.list_user_ids(
            limit=self.config.reconciliation_batch_size,
            offset=self._offset,
        )
        if not user_ids:
            self._offset = 0
            return 0
        if len(user_ids) < self.config.reconciliation_batch_size:
            self._offset = 0
        else:
            self._offset += len(user_ids)

        corrected = 0
        for user_id in user_ids:
            try:
                memory = await self.memories.get(user_id)
                if memory is None:
                    continue
                count = await self.facts_store.count_active(user_id=user_id)
                if count == memory.fact_count:
                    continue
                await self.memories.update_with_retry(
                    user_id,
                    lambda _latest: MemoryChange(fact_count=count),
                )
                corrected += 1
            except Exception:  # noqa: BLE001 - 单用户失败不影响其他用户
                logger.warning("事实对账失败: %s", user_id, exc_info=True)
        return corrected
