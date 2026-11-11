"""参考文献切片：整段一个 chunk，不拆散。"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import DocumentType, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock

from .base import BaseChunker

__all__ = ["ReferenceChunker"]


class ReferenceChunker(BaseChunker):
    section_type = SectionType.REFERENCE

    async def chunk(self, block: ContentBlock, processed: ProcessedBlock, *, kb_id: UUID, doc_type: DocumentType, chunking_version: str, start_index: int = 0) -> list[ChunkDraft]:
        text = (processed.text or block.text).strip()
        if not text:
            return []
        return [self.build_chunk(block=block, processed=processed, kb_id=kb_id, index=start_index, text=text, embedding_text=text, section_type=SectionType.REFERENCE)]
