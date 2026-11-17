"""RAG 入库缓存：chunk 向量与 chunk 级 LLM 描述。

设计目标（《缓存设计》§1、§3）：

- **按内容寻址**：embedding key = 模型 + 维度 + input_type + 文本摘要；
  description key = 模型 + Prompt id/版本/checksum + 请求摘要；
- **重试复用**：入库在 embedding/向量/图/PG 任一阶段失败后重试时，已算过的
  向量和已生成的描述直接命中，不再重复消耗 token；
- **缓存不是权威**：Redis 不可用、值非法、大小超限都按 miss 处理，主流程
  回源继续；
- **熔断式降级**：连续连接失败时短暂停用缓存，避免每个 chunk 都等待超时。

本模块只依赖 ``store.redis`` 的通用缓存原语，不 import RAG 业务模型，
避免缓存层反向依赖入库实现。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from store.redis import CacheKey, CacheScope, RedisCache
from utils.hashing import sha256_text, short_digest
from utils.serialization import canonical_json

__all__ = [
    "ChunkCache",
    "InMemoryChunkCache",
    "ResilientChunkCache",
    "build_default_chunk_cache",
    "description_cache_key",
    "embedding_cache_key",
    "graph_extraction_cache_key",
]

logger = logging.getLogger(__name__)


class ChunkCache(Protocol):
    """RAG 入库缓存的最小协议；Redis/内存/测试 fake 都实现这两个方法。"""

    async def get(self, key: CacheKey) -> Any | None:
        """按结构化 key 读取缓存值；miss/失败返回 None。"""
        ...

    async def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_s: int | None = None,
    ) -> bool:
        """写入缓存值；失败返回 False，不抛穿。"""
        ...


def embedding_cache_key(
    *,
    model_id: str,
    dimensions: int | None,
    input_type: str,
    text: str,
    version: str,
) -> CacheKey:
    """chunk/query 向量的内容寻址 key。

    ``version`` 通常来自 ``IndexerConfig.embedding_version``；模型、维度或
    input_type 任一变化都会得到新 key，旧向量自然失效。
    """
    return CacheKey(
        # query 向量与 chunk 向量分开 scope，便于按不同 TTL/容量策略管理；
        # key 仍按“模型 + 维度 + input_type + 文本摘要”内容寻址。
        scope=(
            CacheScope.QUERY_EMBEDDING
            if input_type == "query"
            else CacheScope.CHUNK_EMBEDDING
        ),
        model_id=model_id,
        query_digest=sha256_text(text),
        version=f"{input_type}:{dimensions or 0}:{version}",
    )


def graph_extraction_cache_key(
    *,
    model_id: str,
    prompt_version: int,
    prompt_checksum: str,
    domain_profile: str,
    text: str,
) -> CacheKey:
    """chunk 级图抽取结果 key。

    key 由 chunk 正文摘要 + 模型 + Prompt 版本/checksum + 领域画像内容签名/运行配置
    组成；重试、REINDEX 或重复上传同一文档时直接复用，不再重复调用 LLM。
    """
    return CacheKey(
        scope=CacheScope.GRAPH_EXTRACTION,
        model_id=model_id,
        version=f"graph:{prompt_version}",
        extra={
            "prompt_checksum": prompt_checksum,
            "domain_profile": domain_profile,
            "chunk": sha256_text(text),
        },
    )


def description_cache_key(
    *,
    model_id: str,
    prompt_id: str,
    prompt_version: int,
    prompt_checksum: str,
    max_chars: int,
    payload: dict[str, Any],
) -> CacheKey:
    """chunk 级 LLM 描述 key。

    请求正文只以规范化 JSON 的短摘要进入 key，Redis key 中不出现原文；
    Prompt 正文改动会改变 ``prompt_checksum``，从而不命中旧描述。
    """
    return CacheKey(
        scope=CacheScope.CHUNK_DESCRIPTION,
        model_id=model_id,
        version=f"{prompt_id}:{prompt_version}:{max_chars}",
        extra={
            "prompt_checksum": prompt_checksum,
            "request": short_digest(canonical_json(payload), 24),
        },
    )


@dataclass
class InMemoryChunkCache:
    """测试/离线运行用缓存；语义与 Redis 版一致，不做过期淘汰。"""

    values: dict[str, Any] = field(default_factory=dict)
    sets: int = 0
    misses: int = 0

    async def get(self, key: CacheKey) -> Any | None:
        """按键读取；miss 返回 None 并计数。"""
        value = self.values.get(key.redis_key())
        if value is None:
            self.misses += 1
        return value

    async def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_s: int | None = None,
    ) -> bool:
        """写入内存字典；``ttl_s`` 保留接口兼容，不做淘汰。"""
        self.values[key.redis_key()] = value
        self.sets += 1
        return True


@dataclass
class ResilientChunkCache:
    """带熔断的缓存包装：Redis 连续失败时短暂停用，避免每 chunk 超时。

    ``RedisCache`` 默认把异常吞成 miss；本包装需要看到异常来决定熔断，因此
    应与 ``CachePolicy(on_error="raise")`` 的 RedisCache 搭配使用。
    """

    cache: ChunkCache
    cooldown_s: float = 30.0
    _disabled_until: float = 0.0
    _disabled_logged: bool = False

    def _available(self) -> bool:
        """当前是否允许访问后端；冷却期内直接返回 False。"""
        return time.monotonic() >= self._disabled_until

    def _disable(self, exc: Exception) -> None:
        """进入冷却期；只在首次失败时打 warning，避免日志刷屏。"""
        self._disabled_until = time.monotonic() + self.cooldown_s
        if not self._disabled_logged:
            logger.warning(
                "RAG 缓存后端不可用，%.0fs 内降级为 miss: %s",
                self.cooldown_s,
                exc,
            )
            self._disabled_logged = True

    async def get(self, key: CacheKey) -> Any | None:
        """读取缓存；后端异常时熔断并返回 miss。"""
        if not self._available():
            return None
        try:
            value = await self.cache.get(key)
        except Exception as exc:  # noqa: BLE001 - 缓存失败不能影响入库
            self._disable(exc)
            return None
        if self._disabled_logged:
            logger.info("RAG 缓存后端已恢复")
            self._disabled_logged = False
        return value

    async def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_s: int | None = None,
    ) -> bool:
        """写入缓存；后端异常时熔断并返回 False。"""
        if not self._available():
            return False
        try:
            return await self.cache.set(key, value, ttl_s=ttl_s)
        except Exception as exc:  # noqa: BLE001 - 缓存失败不能影响入库
            self._disable(exc)
            return False


def build_default_chunk_cache(*, cooldown_s: float = 30.0) -> ChunkCache:
    """构造默认 Redis 缓存：异常上抛给 ``ResilientChunkCache`` 熔断。

    应用未初始化 Redis 时，第一次访问会失败并进入冷却；入库本身继续运行。
    """
    from store.redis import CachePolicy

    return ResilientChunkCache(
        # 单条 3072 维向量的 JSON 约 60KB，默认 64KB 上限容易踩线；
        # chunk 缓存只存单个向量/短描述，放宽到 256KB 仍属于小对象。
        RedisCache(
            policy=CachePolicy(
                on_error="raise",
                max_value_bytes=256 * 1024,
            )
        ),
        cooldown_s=cooldown_s,
    )
