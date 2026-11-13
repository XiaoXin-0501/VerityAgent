"""向量存储接口与 Weaviate DocChunk 实现。

上层只依赖 ``VectorChunkStore`` 协议；真实实现负责：
- chunk → Weaviate properties 映射；
- 批量 upsert（确定性 uuid）；
- 按 doc_id 列出现有 chunk id（重切片时删除 stale）；
- 按 doc_id/chunk_id 删除。
"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.enums import BlockType
from rag.shared.models import Chunk
from store.weaviate import SchemaCheckResult, SchemaManager, WeaviateStore
from exception import WeaviateOperationError
from utils.text import detect_language

__all__ = [
    "VectorChunkStore",
    "WeaviateChunkStore",
    "chunk_content_language",
    "chunk_properties",
]

# 内容语言无法判断时的落库字段；中文项目默认走 zh，避免中文 chunk 落入
# 英文分词器导致 BM25 几乎不可用。
_DEFAULT_CONTENT_LANGUAGE = "zh"

# chunk.language 可能是代码语言（python/tsx…）而不是自然语言；
# 只有明确的自然语言标签才直接采信。
_ZH_LANGUAGE_PREFIXES = ("zh", "cn", "chinese")
_EN_LANGUAGE_PREFIXES = ("en", "english")


class VectorChunkStore(Protocol):
    """向量 chunk 存储协议。"""

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """写入/覆盖 chunk 与向量，返回成功数量。"""
        ...

    async def list_chunk_ids(self, doc_id) -> list[str]:
        """列出某文档当前所有 chunk id，用于重切片删除 stale。"""
        ...

    async def delete_document(self, doc_id) -> int:
        """按 doc_id 删除全部 chunk，幂等；返回删除数。"""
        ...

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """按 chunk_id 删除，幂等。"""
        ...


def chunk_content_language(chunk: Chunk) -> str:
    """判断 chunk 正文应落 ``content_zh`` 还是 ``content_en``。

    顺序：
    1. ``chunk.language`` 是明确的自然语言标签（zh/en/chinese…）→ 直接采信；
    2. 否则对正文做轻量字符占比检测（``utils.text.detect_language``）；
    3. 代码块无法判断时默认 `en`（符号/数字为主），其他默认 `zh`。

    注意：``chunk.language`` 对代码块可能是 ``python/tsx`` 等代码语言，
    不能当作自然语言；此时会走正文检测（代码通常归入 en）。
    """
    declared = (chunk.language or "").strip().lower()
    if declared.startswith(_ZH_LANGUAGE_PREFIXES):
        return "zh"
    if declared.startswith(_EN_LANGUAGE_PREFIXES):
        return "en"
    detected = detect_language(chunk.content or "")
    if detected:
        return detected
    # 符号/数字为主的代码块无法从字符占比判断；默认走英文分词器。
    if chunk.block_type is BlockType.CODE:
        return "en"
    return _DEFAULT_CONTENT_LANGUAGE


def chunk_properties(chunk: Chunk) -> dict:
    """把 Chunk 映射为 DocChunk schema 的 properties。

    正文按语言写入二选一字段：
    - ``content_zh``：中文正文，配 gse_ch 分词器；
    - ``content_en``：英文/代码正文，配 word 分词器；
    - 另一个字段写空字符串（字段始终存在，BM25 查询可按 query 语言选字段）。

    ``heading`` 写最近一级标题，``heading_path`` 写从一级标题到当前的完整
    路径（``text[]``）；``semantic_description`` 写表格/代码的描述（纯文本
    chunk 为空串），便于管理和观察向量化依据；embedding_text 仍由 chunker
    单独生成，不落库。空值不写入，避免可选字段报错。
    """
    heading_path = [item for item in chunk.heading_path if item]
    # Chunk.heading 可能是 chunker 拼好的完整路径字符串；Weaviate 的
    # heading 字段按检索契约存“最近一级”，完整路径由 heading_path 数组承载。
    heading = heading_path[-1] if heading_path else chunk.heading
    language = chunk_content_language(chunk)
    properties = {
        "kb_id": str(chunk.kb_id),
        "doc_id": str(chunk.doc_id),
        "doc_type": chunk.doc_type or "",
        "section_type": chunk.section_type.value,
        "block_type": chunk.block_type.value,
        "heading": heading,
        "heading_path": heading_path or None,
        # 特殊块语义描述；纯文本 chunk 写空串，保证字段始终可观察/可查询。
        "semantic_description": chunk.semantic_description or "",
        "language": language,
        "quality_score": float(chunk.quality_score),
        "chunk_hash": chunk.chunk_hash,
        "index_version": int(chunk.index_version or 0),
        "chunking_version": chunk.chunking_version or "",
        "parser_id": chunk.parser_id or "",
        "chunk_index": int(chunk.chunk_index),
        "page_no": chunk.page_no,
        # 单集合双字段：只有一个字段有正文，另一个为空字符串。
        "content_zh": chunk.content if language == "zh" else "",
        "content_en": chunk.content if language == "en" else "",
        "token_count": int(chunk.token_count),
        "created_at": chunk.created_at,
    }
    return {key: value for key, value in properties.items() if value is not None}


class WeaviateChunkStore:
    """基于 Weaviate ``DocChunk`` 的向量存储实现。"""

    def __init__(self, store: WeaviateStore | None = None) -> None:
        """保存 WeaviateStore；真正的 collection 句柄懒获取。"""
        self.store = store or WeaviateStore()

    async def ensure_schema(
        self,
        *,
        expected_dimensions: int | None = None,
    ) -> SchemaCheckResult:
        """按 import JSON 校验/创建 DocChunk schema（D-97）。"""
        manager = SchemaManager(self.store.config)
        return await manager.check_or_ensure(
            self.store.client,
            expected_dimensions=expected_dimensions,
        )

    async def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> int:
        """批量插入；``chunk_id`` 作为 Weaviate uuid。

        注意：Weaviate 批量 insert 不会覆盖已存在的 uuid。重试/重切片时，
        调用方应先 ``delete_chunks(existing ∩ new_ids)``，再由 pipeline 删除
        stale id（见 ``DefaultIndexingPipeline._index``）。
        """
        if len(chunks) != len(vectors):
            raise WeaviateOperationError(
                "chunk 与向量数量不一致",
                details={"chunks": len(chunks), "vectors": len(vectors)},
            )
        if not chunks:
            return 0
        from weaviate.collections.classes.data import DataObject

        objects = [
            DataObject(
                properties=chunk_properties(chunk),
                uuid=UUID(chunk.chunk_id),
                vector=list(vector),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        result = await self.store.maybe_to_thread(
            self.store.doc_chunks.data.insert_many,
            objects,
        )
        if getattr(result, "has_errors", False):
            raise WeaviateOperationError(
                "Weaviate DocChunk 写入存在失败项",
                details={"errors": getattr(result, "all_responses", None)},
            )
        return len(objects)

    async def list_chunk_ids(self, doc_id) -> list[str]:
        """按 doc_id 列出 chunk id；用于重切片删除 stale。"""
        from weaviate.classes.query import Filter

        result = await self.store.maybe_to_thread(
            self.store.doc_chunks.query.fetch_objects,
            filters=Filter.by_property("doc_id").equal(str(doc_id)),
            limit=10000,
            return_properties=[],
        )
        return [str(item.uuid) for item in getattr(result, "objects", [])]

    async def delete_document(self, doc_id) -> int:
        """按 doc_id 删除，幂等。"""
        result = await self.store.delete_doc_chunks(doc_id)
        return int(result.get("successful", 0))

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        """按 chunk_id 删除；空列表直接返回 0，分批避免过滤条件过大。"""
        if not chunk_ids:
            return 0
        from weaviate.classes.query import Filter

        deleted = 0
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            where = Filter.by_id().contains_any(
                [UUID(chunk_id) for chunk_id in batch]
            )
            result = await self.store.maybe_to_thread(
                self.store.doc_chunks.data.delete_many,
                where=where,
            )
            deleted += int(getattr(result, "successful", 0))
        return deleted

    async def health_check(self) -> dict:
        """健康状态；供 Indexer 启动/健康检查使用。"""
        return await self.store.health_check()
