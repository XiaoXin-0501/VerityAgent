"""图存储接口（v1 由 Neo4j GraphStore 实现）。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import ChunkGraphExtraction

__all__ = ["GraphIndexStore"]


class GraphIndexStore(Protocol):
    async def upsert_chunk(self, chunk_id: str, doc_id: str, kb_id: str, keywords: list[str]) -> None: ...
    async def upsert_relations(self, chunk_id: str, doc_id: str, kb_id: str, relations: list) -> None: ...
    async def delete_document(self, doc_id: str) -> None: ...
