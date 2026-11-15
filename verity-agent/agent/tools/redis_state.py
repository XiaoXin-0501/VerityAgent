"""Redis 工具横切状态后端。

工具流水线只依赖 ``agent.tools.state`` 的窄协议；本文件提供 Redis 实现：

- 结果缓存：短 TTL、按工具声明的 key 精确命中；
- 幂等：``SET NX`` 抢占执行锁，终态覆盖并保留结果；
- 熔断：CLOSED / OPEN / HALF_OPEN 状态机，按窗口统计失败；
- 限流：固定窗口计数，超限返回建议等待时间。

Redis 不可用时，流水线按策略降级：读缓存/熔断/限流 fail-open，写工具幂等
fail-closed（由流水线根据 side_effect 决定）。
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from agent.tools.models import (
    CircuitBreakerRecord,
    IdempotencyRecord,
    ToolResult,
)
from agent.tools.enums import ToolCircuitState
from agent.tools.policies import (
    CircuitBreakerPolicy,
    RateLimitPolicy,
)
from store.redis.connection import RedisConnectionManager, get_redis
from store.redis.keys import RedisKeyPrefix, make_key
from utils.serialization import json_dumps, json_loads
from utils.time import utcnow

__all__ = [
    "RedisToolCache",
    "RedisIdempotencyStore",
    "RedisCircuitBreaker",
    "RedisRateLimiter",
    "build_redis_tool_state",
]

logger = logging.getLogger(__name__)


class RedisToolCache:
    """工具结果缓存；值保存完整 ToolResult 的 JSON。"""

    def __init__(self, redis: RedisConnectionManager | None = None) -> None:
        """保存 Redis 管理器；不在构造期建立连接。"""
        self.redis = redis or get_redis()

    async def get(self, key: str) -> ToolResult | None:
        """读取缓存；缺失/脏值返回 None。"""
        raw = await self.redis.get(key)
        if not raw:
            return None
        try:
            return ToolResult.model_validate(json_loads(raw))
        except Exception:  # noqa: BLE001 - 旧格式/脏值按 miss
            logger.warning("工具缓存值不合法，按 miss 处理: %s", key)
            return None

    async def put(self, key: str, result: ToolResult, ttl_s: int) -> None:
        """写入完整结果；TTL 由工具 CachePolicy 决定。"""
        await self.redis.set(
            key,
            json_dumps(result),
            ex=max(1, int(ttl_s)),
        )


class RedisIdempotencyStore:
    """工具幂等记录 + 执行锁。"""

    def __init__(self, redis: RedisConnectionManager | None = None) -> None:
        """保存 Redis 管理器。"""
        self.redis = redis or get_redis()

    async def get(self, key: str) -> IdempotencyRecord | None:
        """读取幂等记录；缺失/脏值返回 None。"""
        raw = await self.redis.get(key)
        if not raw:
            return None
        try:
            return IdempotencyRecord.model_validate(json_loads(raw))
        except Exception:  # noqa: BLE001 - 脏值按未命中，让写工具重新走锁
            logger.warning("幂等记录不合法，按未命中处理: %s", key)
            return None

    async def acquire(
        self,
        record: IdempotencyRecord,
        *,
        lock_ttl_s: int,
    ) -> bool:
        """原子抢占记录；记录本身按业务 TTL 保留，locked_until 控制执行锁。"""
        # 记录 TTL 取业务过期时间，而不是锁 TTL：锁过期后仍要保留 IN_PROGRESS
        # 证据，供写工具按 UNKNOWN 对账，不能把记录直接删掉。
        ttl_s = max(
            1,
            int((record.expires_at - utcnow()).total_seconds()),
        )
        created = await self.redis.set(
            record.key,
            json_dumps(record),
            ex=ttl_s,
            nx=True,
        )
        if created:
            return True
        # 已有记录时，只有原锁已过期才允许接管；这里保持保守，返回 False，
        # 由流水线读取记录并判定 IN_PROGRESS/UNKNOWN。
        return False

    async def complete(
        self,
        record: IdempotencyRecord,
        *,
        ttl_s: int,
    ) -> None:
        """写入终态（SUCCEEDED/FAILED）并保留结果供重放。"""
        await self.redis.set(
            record.key,
            json_dumps(record),
            ex=max(1, int(ttl_s)),
        )


class RedisCircuitBreaker:
    """按工具维度的 Redis 熔断状态机。"""

    def __init__(self, redis: RedisConnectionManager | None = None) -> None:
        """保存 Redis 管理器。"""
        self.redis = redis or get_redis()

    def _key(self, tool_name: str) -> str:
        """熔断状态 key。"""
        return make_key(RedisKeyPrefix.TOOL_CIRCUIT, tool_name)

    def _ttl_s(self, policy: CircuitBreakerPolicy) -> int:
        """状态保留时间：覆盖统计窗口与冷却期，避免状态过早丢失。"""
        return max(60, policy.window_s * 2, policy.open_cooldown_s * 2)

    async def _load(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> CircuitBreakerRecord:
        """读取状态；缺失/脏值返回新记录。"""
        raw = await self.redis.get(self._key(tool_name))
        if not raw:
            return CircuitBreakerRecord(tool_name=tool_name)
        try:
            record = CircuitBreakerRecord.model_validate(json_loads(raw))
        except Exception:  # noqa: BLE001 - 状态损坏时重新开始
            logger.warning("熔断状态不合法，重置: %s", tool_name)
            record = CircuitBreakerRecord(tool_name=tool_name)
        return self._rotate_window(record, policy)

    async def _save(
        self,
        record: CircuitBreakerRecord,
        policy: CircuitBreakerPolicy,
    ) -> None:
        """写回状态并设置保底 TTL。"""
        await self.redis.set(
            self._key(record.tool_name),
            json_dumps(record),
            ex=self._ttl_s(policy),
        )

    @staticmethod
    def _rotate_window(
        record: CircuitBreakerRecord,
        policy: CircuitBreakerPolicy,
    ) -> CircuitBreakerRecord:
        """统计窗口过期时清零；OPEN/HALF_OPEN 不因窗口轮转而重置。"""
        if record.state is not ToolCircuitState.CLOSED:
            return record
        now = utcnow()
        if (now - record.window_start).total_seconds() >= policy.window_s:
            return record.model_copy(
                update={
                    "failure_count": 0,
                    "success_count": 0,
                    "total_count": 0,
                    "window_start": now,
                }
            )
        return record

    async def allow(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> tuple[bool, float | None]:
        """判断是否放行；OPEN 冷却期后允许有限 HALF_OPEN 探测。"""
        record = await self._load(tool_name, policy)
        now = utcnow()
        if record.state is ToolCircuitState.CLOSED:
            return True, None

        if record.state is ToolCircuitState.OPEN:
            if record.next_probe_at is not None and now >= record.next_probe_at:
                record = record.model_copy(
                    update={
                        "state": ToolCircuitState.HALF_OPEN,
                        "success_count": 0,
                        "half_open_in_flight": 1,
                    }
                )
                await self._save(record, policy)
                return True, None
            retry_after = 1.0
            if record.next_probe_at is not None:
                retry_after = max(
                    0.0,
                    (record.next_probe_at - now).total_seconds(),
                )
            return False, retry_after

        # HALF_OPEN：限制并发探测，避免恢复瞬间打爆下游。
        if record.half_open_in_flight < policy.half_open_max_calls:
            record = record.model_copy(
                update={
                    "half_open_in_flight": record.half_open_in_flight + 1,
                }
            )
            await self._save(record, policy)
            return True, None
        return False, 1.0

    async def record_success(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> None:
        """记录成功；HALF_OPEN 连续成功后回到 CLOSED。"""
        record = await self._load(tool_name, policy)
        if record.state is ToolCircuitState.HALF_OPEN:
            success_count = record.success_count + 1
            if success_count >= policy.success_threshold:
                record = record.model_copy(
                    update={
                        "state": ToolCircuitState.CLOSED,
                        "failure_count": 0,
                        "success_count": 0,
                        "total_count": 0,
                        "half_open_in_flight": 0,
                        "opened_at": None,
                        "next_probe_at": None,
                        "window_start": utcnow(),
                    }
                )
            else:
                record = record.model_copy(
                    update={
                        "success_count": success_count,
                        "total_count": record.total_count + 1,
                        "half_open_in_flight": max(
                            0, record.half_open_in_flight - 1
                        ),
                    }
                )
        else:
            record = record.model_copy(
                update={"total_count": record.total_count + 1}
            )
        await self._save(record, policy)

    async def record_failure(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> None:
        """记录失败；达到次数或失败率阈值时进入 OPEN。"""
        record = await self._load(tool_name, policy)
        total = record.total_count + 1
        failures = record.failure_count + 1
        if record.state is ToolCircuitState.HALF_OPEN:
            record = record.model_copy(
                update={
                    "state": ToolCircuitState.OPEN,
                    "failure_count": failures,
                    "total_count": total,
                    "success_count": 0,
                    "half_open_in_flight": max(
                        0, record.half_open_in_flight - 1
                    ),
                    "opened_at": utcnow(),
                    "next_probe_at": utcnow()
                    + timedelta(seconds=policy.open_cooldown_s),
                }
            )
            await self._save(record, policy)
            return

        opened = failures >= policy.failure_threshold
        if policy.failure_rate_threshold is not None and total > 0:
            opened = opened or (
                failures / total >= policy.failure_rate_threshold
            )
        updates: dict[str, Any] = {
            "failure_count": failures,
            "total_count": total,
        }
        if opened:
            now = utcnow()
            updates.update(
                {
                    "state": ToolCircuitState.OPEN,
                    "opened_at": now,
                    "next_probe_at": now
                    + timedelta(seconds=policy.open_cooldown_s),
                    "success_count": 0,
                    "half_open_in_flight": 0,
                }
            )
        await self._save(record.model_copy(update=updates), policy)


class RedisRateLimiter:
    """按工具维度的固定窗口限流。"""

    def __init__(self, redis: RedisConnectionManager | None = None) -> None:
        """保存 Redis 管理器。"""
        self.redis = redis or get_redis()

    async def acquire(
        self,
        key: str,
        policy: RateLimitPolicy,
    ) -> tuple[bool, float | None]:
        """INCR 计数并在窗口首次调用时设置过期时间。"""
        if policy.max_calls is None:
            return True, None
        count = int(await self.redis.incr(key))
        if count == 1:
            await self.redis.expire(key, policy.window_s)
        if count <= policy.max_calls:
            return True, None
        ttl = await self.redis.ttl(key)
        return False, float(ttl if ttl and ttl > 0 else policy.window_s)


def build_redis_tool_state(
    redis: RedisConnectionManager | None = None,
) -> dict[str, Any]:
    """构造一组共享同一 Redis 管理器的工具状态后端。"""
    manager = redis or get_redis()
    return {
        "cache": RedisToolCache(manager),
        "idempotency": RedisIdempotencyStore(manager),
        "circuit_breaker": RedisCircuitBreaker(manager),
        "rate_limiter": RedisRateLimiter(manager),
    }
