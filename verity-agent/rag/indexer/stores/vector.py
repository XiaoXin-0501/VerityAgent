"""向量存储接口（v1 由 Weaviate DocChunk 实现）。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import Chunk

__all__ = ["VectorChunkStore"]


class VectorChunkStore(Protocol):
    async def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> int: ...
    async def delete_document(self, doc_id) -> int: ...
    async def delete_chunks(self, chunk_ids: list[str]) -> int: ...
