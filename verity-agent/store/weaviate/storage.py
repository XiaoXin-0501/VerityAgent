"""Weaviate 集合门面。

只放集合级通用操作：取 collection、按属性删除、健康状态。DocChunk / UserFact
的具体读写、查询构造由 RAG / 记忆模块实现。
"""

from __future__ import annotations

import asyncio
from typing import Any

from exception import WeaviateOperationError

from .connection import WeaviateConnectionManager, get_weaviate

__all__ = ["WeaviateStore"]


class WeaviateStore:
    def __init__(self, manager: WeaviateConnectionManager | None = None) -> None:
        self._manager = manager

    @property
    def manager(self) -> WeaviateConnectionManager:
        return self._manager or get_weaviate()

    @property
    def config(self):
        return self.manager.config

    @property
    def client(self):
        return self.manager.client

    def collection(self, name: str):
        """取同步 collection 句柄；调用其方法时用 ``maybe_to_thread``。"""
        return self.client.collections.get(name)

    @property
    def doc_chunks(self):
        return self.collection(self.config.doc_chunk_collection)

    @property
    def user_facts(self):
        return self.collection(self.config.user_fact_collection)

    async def maybe_to_thread(self, func, *args, **kwargs):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as exc:
            raise WeaviateOperationError(
                "Weaviate 操作失败", details={"cause": str(exc)}
            ) from exc

    async def delete_doc_chunks(
        self,
        doc_id: Any,
        *,
        kb_id: Any | None = None,
    ) -> dict[str, int]:
        """按 doc_id（可选 kb_id）删除 DocChunk，幂等。"""
        from weaviate.classes.query import Filter

        where = Filter.by_property("doc_id").equal(str(doc_id))
        if kb_id is not None:
            where = where & Filter.by_property("kb_id").equal(str(kb_id))
        result = await self.maybe_to_thread(
            self.doc_chunks.data.delete_many, where=where
        )
        return {
            "matches": int(getattr(result, "matches", 0)),
            "successful": int(getattr(result, "successful", 0)),
            "failed": int(getattr(result, "failed", 0)),
        }

    async def delete_user_facts(self, user_id: Any) -> dict[str, int]:
        """按 user_id 删除 UserFact；用户删除状态机调用。"""
        from weaviate.classes.query import Filter

        where = Filter.by_property("user_id").equal(str(user_id))
        result = await self.maybe_to_thread(
            self.user_facts.data.delete_many, where=where
        )
        return {
            "matches": int(getattr(result, "matches", 0)),
            "successful": int(getattr(result, "successful", 0)),
            "failed": int(getattr(result, "failed", 0)),
        }

    async def health_check(self) -> dict[str, Any]:
        return await self.manager.health_check()
