"""Redis 实时用量计数（best-effort，不是权威账本）。"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from config.hooks import TokenUsageHookConfig
from exception import RedisError
from store.redis import RedisConnectionManager, get_redis, make_key
from store.redis.keys import RedisKeyPrefix
from utils.time import today_utc, utcnow

from .enum import UsageCounterScope
from .models import UsageCounter, UsageRecord

logger = logging.getLogger(__name__)

__all__ = ["UsageCounterStore"]


class UsageCounterStore:
    def __init__(
        self,
        config: TokenUsageHookConfig,
        *,
        redis: RedisConnectionManager | None = None,
    ) -> None:
        """保存计数配置与可选 Redis 客户端。"""
        self.config = config
        self._redis = redis

    @property
    def redis(self) -> RedisConnectionManager:
        """返回注入的 Redis 客户端或全局实例。"""
        return self._redis or get_redis()

    @property
    def scopes(self) -> list[UsageCounterScope]:
        """解析配置中的计数维度，忽略未知值。"""
        result: list[UsageCounterScope] = []
        for value in self.config.counter_scopes:
            try:
                result.append(UsageCounterScope(value))
            except ValueError:
                logger.warning("未知 UsageCounterScope: %s", value)
        return result

    async def seen(self, call_id: str) -> bool:
        """返回 True 表示第一次见到该 call_id；Redis 失败时按未见过处理。"""
        if not call_id:
            return True
        key = make_key(RedisKeyPrefix.USAGE, "seen", call_id)
        try:
            result = await self.redis.set(
                key, "1", ex=self.config.dedupe_ttl_s, nx=True
            )
            return bool(result)
        except RedisError as exc:
            logger.warning("Redis usage seen 写入失败: %s", exc)
            return True

    async def record(
        self,
        record: UsageRecord,
        *,
        user_id: UUID | None = None,
    ) -> list[UsageCounter]:
        """按配置维度累加用量并返回本次计数快照。"""
        counters: list[UsageCounter] = []
        for scope in self.scopes:
            identifier = self._identifier(scope, record, user_id=user_id)
            if identifier is None:
                continue
            key = make_key(RedisKeyPrefix.USAGE, scope.value, identifier)
            try:
                await self._write_hash(key, record)
                ttl = self.config.counter_ttl_s.get(scope.value)
                if ttl:
                    await self.redis.expire(key, int(ttl))
                counters.append(
                    UsageCounter(scope=scope, key=key, **self._counter_fields(record))
                )
            except RedisError as exc:
                logger.warning("Redis 用量计数失败: scope=%s key=%s %s", scope, key, exc)
        return counters

    async def read(self, scope: UsageCounterScope, identifier: str) -> UsageCounter | None:
        """读取指定维度与标识的累计计数。"""
        key = make_key(RedisKeyPrefix.USAGE, scope.value, identifier)
        try:
            data = await self.redis.hgetall(key)
        except RedisError:
            return None
        if not data:
            return None
        return UsageCounter(
            scope=scope,
            key=key,
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            cached_input_tokens=int(data.get("cached_input_tokens", 0)),
            reasoning_tokens=int(data.get("reasoning_tokens", 0)),
            total_cost=float(data.get("total_cost", 0.0)),
            call_count=int(data.get("call_count", 0)),
        )

    async def _write_hash(self, key: str, record: UsageRecord) -> None:
        """把用量字段累加写入 Redis hash。"""
        usage = record.usage
        fields = {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "call_count": 1,
        }
        for name, value in fields.items():
            await self.redis.hincrby(key, name, int(value))
        if record.cost is not None:
            await self.redis.hincrbyfloat(key, "total_cost", record.cost.total_cost)
        await self.redis.hset(key, mapping={"updated_at": utcnow().isoformat()})

    @staticmethod
    def _counter_fields(record: UsageRecord) -> dict[str, Any]:
        """把 UsageRecord 转换为计数字段。"""
        usage = record.usage
        return {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cached_input_tokens": usage.cached_input_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
            "total_cost": record.cost.total_cost if record.cost else 0.0,
            "call_count": 1,
        }

    def _identifier(
        self,
        scope: UsageCounterScope,
        record: UsageRecord,
        *,
        user_id: UUID | None,
    ) -> str | None:
        """按维度生成计数标识；无法生成时返回 None。"""
        if scope is UsageCounterScope.TURN:
            return record.turn_id
        if scope is UsageCounterScope.CONVERSATION:
            return str(record.conversation_id) if record.conversation_id else None
        if user_id is None:
            return None
        return f"{user_id}:{today_utc().isoformat()}"
