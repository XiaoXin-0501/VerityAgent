"""切片器协议与公共辅助。"""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, QualityFlag, SectionType
from rag.shared.ids import chunk_hash, deterministic_chunk_id
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from utils.tokens import estimate_tokens

__all__ = ["Chunker", "BaseChunker", "section_type_for"]

_SECTION_TYPES = {
    BlockType.TABLE: SectionType.TABLE,
    BlockType.CODE: SectionType.CODE,
    BlockType.FIGURE: SectionType.FIGURE,
    BlockType.LIST: SectionType.LIST,
    BlockType.FORMULA: SectionType.FORMULA,
    BlockType.REFERENCE: SectionType.REFERENCE,
}


def section_type_for(block_type: BlockType) -> SectionType:
    return _SECTION_TYPES.get(block_type, SectionType.PARAGRAPH)


class BaseChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 512,
        max_tokens: int = 800,
        min_tokens: int = 40,
        overlap_tokens: int = 80,
        chunking_version: str = "v1",
        chunk_id_namespace: str = "verity:chunk",
    ) -> None:
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.chunking_version = chunking_version
        self.chunk_id_namespace = chunk_id_namespace

    def build_chunk(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        index: int,
        text: str,
        embedding_text: str | None = None,
        section_type: SectionType | None = None,
        quality_flags: set[QualityFlag] | None = None,
        metadata: dict | None = None,
    ) -> ChunkDraft:
        body = text.strip()
        digest = chunk_hash(body)
        mode = section_type or section_type_for(block.block_type)
        return ChunkDraft(
            chunk_id=deterministic_chunk_id(
                kb_id=kb_id,
                doc_id=block.doc_id,
                chunk_index=index,
                content_hash=digest,
                chunking_version=self.chunking_version,
                namespace=self.chunk_id_namespace,
            ),
            doc_id=block.doc_id,
            kb_id=kb_id,
            chunk_index=index,
            block_ids=[block.block_id],
            block_type=block.block_type,
            section_type=mode,
            text=body,
            embedding_text=(embedding_text or body).strip(),
            token_count=estimate_tokens(body),
            page_no=block.page_no or None,
            heading=block.heading_text,
            language=block.language,
            quality_flags=quality_flags or set(processed.quality_flags),
            metadata=metadata or {},
        )

    def embedding_text(self, block: ContentBlock, text: str) -> str:
        if block.heading_text:
            return f"{block.heading_text}\n{text}"
        return text


class Chunker(Protocol):
    async def chunk(
        self,
        block: ContentBlock,
        processed: ProcessedBlock,
        *,
        kb_id: UUID,
        doc_type: DocumentType,
        chunking_version: str,
        start_index: int = 0,
    ) -> list[ChunkDraft]:
        """把一个块切成一个或多个 ChunkDraft；不跨块类型。"""
        ...
