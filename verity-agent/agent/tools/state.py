"""工具横切状态的后端协议。

工具模块不直接依赖 Redis 客户端，只依赖这些窄协议；生产实现放
``store.redis``，测试可用内存 fake。v1 不允许把进程内状态作为生产默认值。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from agent.message.models import ToolFile

from .models import IdempotencyRecord, ToolResult
from .policies import CircuitBreakerPolicy, RateLimitPolicy

__all__ = [
    "ToolCacheStore",
    "IdempotencyBackend",
    "CircuitBreakerBackend",
    "RateLimitBackend",
    "OverflowWriter",
]


class ToolCacheStore(Protocol):
    """工具结果短期缓存。"""

    async def get(self, key: str) -> ToolResult | None:
        """未命中返回 None。"""
        ...

    async def put(self, key: str, result: ToolResult, ttl_s: int) -> None:
        """写入缓存；值应包含完整 ToolResult。"""
        ...


class IdempotencyBackend(Protocol):
    """幂等记录 + 锁。"""

    async def get(self, key: str) -> IdempotencyRecord | None:
        """读取幂等记录；不存在返回 None。"""
        ...

    async def acquire(
        self,
        record: IdempotencyRecord,
        *,
        lock_ttl_s: int,
    ) -> bool:
        """原子抢占 IN_PROGRESS 锁；已存在且未过期时返回 False。"""
        ...

    async def complete(
        self,
        record: IdempotencyRecord,
        *,
        ttl_s: int,
    ) -> None:
        """写入终态（SUCCEEDED/FAILED/UNKNOWN）并释放锁。"""
        ...


class CircuitBreakerBackend(Protocol):
    """按工具维度的熔断状态。"""

    async def allow(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> tuple[bool, float | None]:
        """是否允许调用；不允许时返回建议等待秒数。"""
        ...

    async def record_success(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> None: ...

    async def record_failure(
        self,
        tool_name: str,
        policy: CircuitBreakerPolicy,
    ) -> None: ...


class RateLimitBackend(Protocol):
    """按工具维度的限流计数。"""

    async def acquire(
        self,
        key: str,
        policy: RateLimitPolicy,
    ) -> tuple[bool, float | None]:
        """是否放行；拒绝时返回建议等待秒数。"""
        ...


class OverflowWriter(Protocol):
    """大结果溢出写 MinIO。"""

    async def write(
        self,
        *,
        conversation_id: UUID,
        call_id: str,
        tool_name: str,
        content: str,
        data: object,
    ) -> ToolFile:
        """返回可放入 ToolResult.overflow_file 的引用。"""
        ...
