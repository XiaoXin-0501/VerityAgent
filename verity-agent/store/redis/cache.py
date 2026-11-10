"""Redis 缓存原语（《缓存设计》§2–§3）。

只提供“存什么、key 怎么建、TTL/失效”的通用能力；具体缓存值（ToolResult、
RagRetrieval、MemoryProfile 等）由各模块定义，避免这里反向依赖业务模型。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from exception import RedisError, ValidationError
from utils.serialization import json_dumps, json_loads

from .connection import RedisConnectionManager, get_redis
from .keys import RedisKeyPrefix, make_key

__all__ = [
    "CacheScope",
    "CachePolicy",
    "CacheKey",
    "CacheEntry",
    "CacheStats",
    "RedisCache",
    "make_cache_key",
]


class CacheScope(StrEnum):
    TOOL_RESULT = "tool"
    RAG_RETRIEVAL = "rag:retrieval"
    QUERY_EMBEDDING = "rag:embedding"
    RERANK = "rag:rerank"
    GRAPH_NEXT_HOP = "graph"
    MEMORY_PROFILE = "memory:profile"
    MEMORY_SUMMARY = "memory:summary"
    MEMORY_FACTS = "memory:facts"
    DOC_META = "doc_meta"


# 各 scope 的默认 TTL（秒）；可在 CachePolicy 覆盖
DEFAULT_TTLS: dict[CacheScope, int] = {
    CacheScope.TOOL_RESULT: 300,
    CacheScope.RAG_RETRIEVAL: 300,
    CacheScope.QUERY_EMBEDDING: 86_400,
    CacheScope.RERANK: 3_600,
    CacheScope.GRAPH_NEXT_HOP: 3_600,
    CacheScope.MEMORY_PROFILE: 600,
    CacheScope.MEMORY_SUMMARY: 600,
    CacheScope.MEMORY_FACTS: 300,
    CacheScope.DOC_META: 3_600,
}


class CachePolicy(BaseModel):
    """缓存策略；按 scope 覆盖默认 TTL。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # 是否启用该缓存
    enabled: bool = True
    # 后端类型；v1 固定 redis（Prompt cache 不属于 Redis）
    backend: str = "redis"
    # TTL（秒）；None 时使用 scope 默认值
    ttl_s: int | None = None
    # 单值大小上限；超过直接跳过缓存
    max_value_bytes: int = Field(default=64 * 1024, gt=0)
    # 参与 key 的字段；None 表示全部
    key_fields: list[str] | None = None
    # 不参与 key 的字段（如耗时/展示字段）
    exclude_fields: list[str] = Field(default_factory=list)
    # key 是否带权限摘要，防止跨用户命中
    include_permission: bool = True
    # 版本来源字段名，如 knowledge_bases.index_version
    version_source: str | None = None
    # Redis 错误策略：miss（回源）/ raise
    on_error: str = "miss"
    # 序列化方式：json / msgpack（v1 只实现 json）
    serialization: str = "json"
    # TTL 随机抖动比例，防雪崩
    ttl_jitter: float = Field(default=0.1, ge=0.0, le=1.0)

    def effective_ttl(self, scope: CacheScope) -> int:
        return self.ttl_s if self.ttl_s is not None else DEFAULT_TTLS[scope]


class CacheKey(BaseModel):
    """缓存 key 的结构化描述；最终 key 只包含 digest/版本，不包含明文正文。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: CacheScope
    user_id: UUID | None = None
    conversation_id: UUID | None = None
    resource_ids: list[str] = Field(default_factory=list)
    query_digest: str | None = None
    permission_digest: str | None = None
    version: str | None = None
    model_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def digest(self) -> str:
        """最终 key 的稳定 SHA256；用于日志/审计，不直接作为 Redis key。"""
        from utils.hashing import sha256_text

        return sha256_text(self.redis_key())

    def redis_key(self) -> str:
        parts: list[str] = [self.scope.value]
        if self.user_id is not None:
            parts.append(str(self.user_id))
        if self.conversation_id is not None:
            parts.append(str(self.conversation_id))
        if self.resource_ids:
            parts.append(",".join(sorted(self.resource_ids)))
        if self.query_digest:
            parts.append(self.query_digest)
        if self.permission_digest:
            parts.append(self.permission_digest)
        if self.model_id:
            parts.append(self.model_id)
        if self.version:
            parts.append(self.version)
        if self.extra:
            # 额外过滤条件先规范化再放最后一段，避免 key 爆炸
            from utils.hashing import short_digest
            from utils.serialization import canonical_json

            parts.append(short_digest(canonical_json(self.extra), 16))
        return make_key(RedisKeyPrefix.CACHE, *parts)


def make_cache_key(scope: CacheScope, *parts: Any) -> str:
    """直接拼 cache key；``parts`` 由调用方保证已规范化/哈希。"""
    return make_key(RedisKeyPrefix.CACHE, scope.value, *parts)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    key: str
    scope: CacheScope
    value: Any
    value_type: str
    created_at: datetime
    expires_at: datetime | None = None
    version: str | None = None
    size_bytes: int = 0


@dataclass
class CacheStats:
    # 命中次数
    hits: int = 0
    # 未命中次数
    misses: int = 0
    # Redis 错误次数（按策略降级也算）
    errors: int = 0
    # 写入次数
    sets: int = 0
    # 淘汰次数（由外部统计填充）
    evictions: int = 0
    # 当前缓存值大小估算
    size_bytes: int = 0
    # 估算因命中节省的延迟（毫秒）
    latency_saved_ms: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class RedisCache:
    """Redis JSON 缓存薄封装。

    - Redis 错误默认按 cache miss 处理，回源业务；``on_error=raise`` 时上抛；
    - TTL 加随机抖动，防同一时刻大面积过期；
    - 单值超过 ``max_value_bytes`` 直接跳过，不缓存大对象。
    """

    def __init__(
        self,
        redis: RedisConnectionManager | None = None,
        *,
        policy: CachePolicy | None = None,
    ) -> None:
        self._redis = redis
        self._policy = policy or CachePolicy()
        self._stats: dict[CacheScope, CacheStats] = {}

    @property
    def redis(self) -> RedisConnectionManager:
        return self._redis or get_redis()

    def stats(self, scope: CacheScope) -> CacheStats:
        return self._stats.setdefault(scope, CacheStats())

    async def get(
        self,
        key: str | CacheKey,
        *,
        scope: CacheScope | None = None,
        policy: CachePolicy | None = None,
        expected_version: str | None = None,
    ) -> Any | None:
        cfg = policy or self._policy
        if not cfg.enabled:
            return None
        redis_key = key.redis_key() if isinstance(key, CacheKey) else key
        actual_scope = scope or (key.scope if isinstance(key, CacheKey) else None)
        try:
            raw = await self.redis.get(redis_key)
        except RedisError:
            if cfg.on_error == "raise":
                raise
            if actual_scope:
                self.stats(actual_scope).errors += 1
            return None
        if raw is None:
            if actual_scope:
                self.stats(actual_scope).misses += 1
            return None
        try:
            entry = json_loads(raw)
            if expected_version is not None and entry.get("version") != expected_version:
                if actual_scope:
                    self.stats(actual_scope).misses += 1
                return None
            if actual_scope:
                st = self.stats(actual_scope)
                st.hits += 1
                st.size_bytes = max(st.size_bytes, int(entry.get("size_bytes", 0)))
            return entry.get("value")
        except (TypeError, ValueError):
            if cfg.on_error == "raise":
                raise
            if actual_scope:
                self.stats(actual_scope).errors += 1
            return None

    async def set(
        self,
        key: str | CacheKey,
        value: Any,
        *,
        scope: CacheScope | None = None,
        policy: CachePolicy | None = None,
        version: str | None = None,
        ttl_s: int | None = None,
    ) -> bool:
        cfg = policy or self._policy
        if not cfg.enabled:
            return False
        redis_key = key.redis_key() if isinstance(key, CacheKey) else key
        actual_scope = scope or (key.scope if isinstance(key, CacheKey) else None)
        payload = json_dumps(
            {
                "value": value,
                "value_type": type(value).__name__,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "version": version,
                "size_bytes": len(json_dumps(value).encode("utf-8")),
            }
        )
        size = len(payload.encode("utf-8"))
        if size > cfg.max_value_bytes:
            return False
        effective_ttl = ttl_s or (cfg.effective_ttl(actual_scope) if actual_scope else 300)
        effective_ttl = max(1, int(effective_ttl * (1 + random.uniform(0, cfg.ttl_jitter))))
        try:
            await self.redis.set(redis_key, payload, ex=effective_ttl)
        except RedisError:
            if cfg.on_error == "raise":
                raise
            if actual_scope:
                self.stats(actual_scope).errors += 1
            return False
        if actual_scope:
            st = self.stats(actual_scope)
            st.sets += 1
            st.size_bytes = max(st.size_bytes, size)
        return True

    async def get_or_load(
        self,
        key: CacheKey,
        loader: Callable[[], Awaitable[Any]],
        *,
        policy: CachePolicy | None = None,
        version: str | None = None,
        ttl_s: int | None = None,
    ) -> Any:
        cached = await self.get(key, policy=policy, expected_version=version)
        if cached is not None:
            return cached
        value = await loader()
        await self.set(key, value, policy=policy, version=version, ttl_s=ttl_s)
        return value

    async def delete(self, key: str | CacheKey) -> bool:
        redis_key = key.redis_key() if isinstance(key, CacheKey) else key
        return await self.redis.delete(redis_key) > 0

    async def delete_many(self, keys: list[str | CacheKey]) -> int:
        redis_keys = [k.redis_key() if isinstance(k, CacheKey) else k for k in keys]
        if not redis_keys:
            return 0
        return await self.redis.delete(*redis_keys)
