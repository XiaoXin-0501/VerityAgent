"""事实置信度惩罚队列与 Worker。"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from store.redis import RedisConnectionManager, get_redis
from store.redis.keys import RedisKeyPrefix, make_key
from utils.serialization import json_dumps, json_loads
from utils.time import utcnow

__all__ = [
    "FactConfidencePenalty",
    "ConflictPenaltyQueue",
    "FactConfidencePenaltyWorker",
]

logger = logging.getLogger(__name__)


class _Model(BaseModel):
    """惩罚队列消息基类。"""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class FactConfidencePenalty(_Model):
    """一次事实置信度下调请求。"""

    user_id: UUID
    fact_ids: list[str] = Field(default_factory=list)
    delta: float = Field(default=0.1, gt=0.0, le=1.0)
    reason: str = ""
    chunk_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    # 已失败重试次数；worker 据此决定是否重新入队
    attempts: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utcnow)


class ConflictPenaltyQueue:
    """基于 Redis List 的惩罚队列；生产端可来自 resolve_chunk_conflict。"""

    def __init__(
        self,
        redis: RedisConnectionManager | None = None,
        *,
        key: str | None = None,
        max_len: int = 10_000,
    ) -> None:
        """保存 Redis 管理器、队列 key 与最大长度。"""
        self.redis = redis or get_redis()
        self.key = key or make_key(RedisKeyPrefix.QUEUE, "memory", "penalty")
        self.max_len = max(1, int(max_len))

    async def enqueue(self, penalty: FactConfidencePenalty) -> None:
        """RPUSH 一条惩罚请求，并裁剪队列长度防止无限增长。"""
        await self.redis.execute_command(
            "RPUSH",
            self.key,
            json_dumps(penalty),
        )
        # 保留最新 max_len 条；LPOP 仍从最旧开始消费，语义保持 FIFO。
        await self.redis.execute_command("LTRIM", self.key, -self.max_len, -1)

    async def dequeue(self, limit: int = 20) -> list[FactConfidencePenalty]:
        """逐条 LPOP；坏值跳过，避免毒消息阻塞队列。"""
        values: list[FactConfidencePenalty] = []
        for _ in range(max(1, int(limit))):
            raw = await self.redis.execute_command("LPOP", self.key)
            if raw is None:
                break
            try:
                values.append(
                    FactConfidencePenalty.model_validate(json_loads(raw))
                )
            except Exception:  # noqa: BLE001 - 毒消息丢弃
                logger.warning("惩罚队列消息不合法，已跳过")
        return values


class FactConfidencePenaltyWorker:
    """消费惩罚队列，下调 UserFact confidence。"""

    def __init__(
        self,
        *,
        queue: ConflictPenaltyQueue,
        facts_store,
        disputed_threshold: float = 0.4,
        batch_size: int = 20,
        interval_s: float = 5.0,
        max_attempts: int = 3,
    ) -> None:
        """保存队列、事实存储与策略。"""
        self.queue = queue
        self.facts_store = facts_store
        self.disputed_threshold = disputed_threshold
        self.batch_size = max(1, int(batch_size))
        self.interval_s = max(0.1, float(interval_s))
        self.max_attempts = max(1, int(max_attempts))
        self._stop = asyncio.Event()

    def stop(self) -> None:
        """请求 worker 退出。"""
        self._stop.set()

    async def run_forever(self) -> None:
        """轮询队列；空队列等待 interval。"""
        while not self._stop.is_set():
            try:
                worked = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - 单轮失败继续
                logger.exception("事实惩罚 worker 异常")
                worked = False
            if worked:
                continue
            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self.interval_s,
                )
            except asyncio.TimeoutError:
                continue

    async def run_once(self) -> bool:
        """处理一批惩罚；返回是否处理了消息。"""
        penalties = await self.queue.dequeue(self.batch_size)
        for penalty in penalties:
            try:
                await self.facts_store.penalize_facts(
                    user_id=penalty.user_id,
                    fact_ids=penalty.fact_ids,
                    delta=penalty.delta,
                    disputed_threshold=self.disputed_threshold,
                )
            except Exception:  # noqa: BLE001 - 单条失败不影响其他
                logger.warning(
                    "事实置信度惩罚失败: %s", penalty.user_id, exc_info=True
                )
                # 失败消息重新入队；超过上限后丢弃并记录，避免毒消息阻塞队列。
                next_attempt = penalty.attempts + 1
                if next_attempt < self.max_attempts:
                    try:
                        await self.queue.enqueue(
                            penalty.model_copy(update={"attempts": next_attempt})
                        )
                    except Exception:  # noqa: BLE001 - 入队失败只能丢弃
                        logger.warning(
                            "惩罚消息重新入队失败: %s",
                            penalty.user_id,
                            exc_info=True,
                        )
                else:
                    logger.error(
                        "事实置信度惩罚超过重试上限，已丢弃: user=%s facts=%s",
                        penalty.user_id,
                        penalty.fact_ids,
                    )
        return bool(penalties)
