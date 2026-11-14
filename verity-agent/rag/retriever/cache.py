"""检索缓存 key 构造（Redis 缓存策略在 store.redis.cache）。"""

from __future__ import annotations

from typing import Any, Protocol

from store.redis.cache import CacheKey, CacheScope
from utils.hashing import sha256_text, short_digest
from utils.serialization import canonical_json

__all__ = [
    "RetrievalCache",
    "retrieval_cache_key",
    "query_digest",
    "permission_digest",
    "filters_digest",
]


class RetrievalCache(Protocol):
    """检索缓存最小协议；与入库缓存结构一致，便于测试注入 fake。"""

    async def get(self, key: CacheKey) -> Any | None:
        """按结构化 key 读取；miss/失败返回 None。"""
        ...

    async def set(
        self,
        key: CacheKey,
        value: Any,
        *,
        ttl_s: int | None = None,
    ) -> bool:
        """写入缓存；失败返回 False，不抛穿。"""
        ...


def query_digest(query: str) -> str:
    """query 的稳定摘要，缓存 key 不落明文 query。"""
    return sha256_text(query)


def permission_digest(permissions: frozenset[str] | set[str]) -> str:
    """权限快照摘要；同 query 不同权限不能共享检索缓存。"""
    return short_digest("|".join(sorted(permissions)), 32)


def filters_digest(filters: Any | None) -> str:
    """过滤条件摘要；Pydantic 模型转规范化 JSON，None 使用固定值。"""
    if filters is None:
        return "none"
    if hasattr(filters, "model_dump"):
        payload = filters.model_dump(mode="json")
    else:
        payload = filters
    return short_digest(canonical_json(payload), 32)


def retrieval_cache_key(
    *,
    query: str,
    kb_ids: list[str],
    permission_digest: str,
    index_version: str,
    embedding_model: str,
    rerank_model: str | None,
    top_k: int,
    filters_digest: str | None = None,
) -> CacheKey:
    """构造检索结果缓存 key。

    key 必须包含：query 摘要、KB 列表、权限摘要、索引版本、embedding/rerank
    模型版本、top_k 与过滤条件摘要；任一变化都会得到新 key，避免跨权限或
    跨索引版本命中旧结果（D-84）。
    """
    return CacheKey(
        scope=CacheScope.RAG_RETRIEVAL,
        resource_ids=sorted(kb_ids),
        query_digest=query_digest(query),
        permission_digest=permission_digest,
        version=short_digest(
            "|".join(
                [
                    index_version,
                    embedding_model,
                    rerank_model or "",
                    str(top_k),
                    filters_digest or "",
                ]
            ),
            32,
        ),
    )
