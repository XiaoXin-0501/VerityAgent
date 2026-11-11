"""检索缓存 key 构造（Redis 缓存策略在 store.redis.cache）。"""

from __future__ import annotations

from store.redis.cache import CacheKey, CacheScope
from utils.hashing import sha256_text, short_digest

__all__ = ["retrieval_cache_key", "query_digest"]


def query_digest(query: str) -> str:
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
