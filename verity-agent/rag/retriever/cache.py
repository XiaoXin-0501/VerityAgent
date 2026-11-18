"""检索缓存 key 构造（Redis 缓存策略在 store.redis.cache）。"""

from __future__ import annotations

from typing import Any

from store.redis.cache import CacheKey, CachePort, CacheScope
from utils.hashing import sha256_text, short_digest
from utils.serialization import canonical_json

__all__ = [
    "RetrievalCache",
    "retrieval_cache_key",
    "query_digest",
    "permission_digest",
    "filters_digest",
    "document_status_cache_key",
    "document_meta_cache_key",
]


# 检索缓存与入库缓存使用同一个最小协议；保留旧名字兼容调用方。
RetrievalCache = CachePort


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


def document_status_cache_key(
    *,
    doc_id: str,
    index_version: int,
) -> CacheKey:
    """文档状态 key：doc_id + KB 索引版本；重切片/提交后自然失效。"""
    return CacheKey(
        scope=CacheScope.DOC_STATUS,
        resource_ids=[doc_id],
        version=str(index_version),
    )


def document_meta_cache_key(
    *,
    doc_id: str,
    index_version: int,
) -> CacheKey:
    """DocumentMeta key：doc_id + KB 索引版本；描述更新随索引版本失效。"""
    return CacheKey(
        scope=CacheScope.DOC_META,
        resource_ids=[doc_id],
        version=str(index_version),
    )


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
