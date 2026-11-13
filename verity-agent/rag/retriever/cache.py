"""检索缓存 key 构造（Redis 缓存策略在 store.redis.cache）。"""

from __future__ import annotations

from store.redis.cache import CacheKey, CacheScope
from utils.hashing import sha256_text, short_digest

__all__ = ["retrieval_cache_key", "query_digest"]


def query_digest(query: str) -> str:
    """query 的稳定摘要，缓存 key 不落明文 query。"""
    return sha256_text(query)


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
