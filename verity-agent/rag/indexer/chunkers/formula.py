"""公式切片：LaTeX + 前后解释，不拆公式。"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import DocumentType, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock

from .base import BaseChunker

__all__ = ["FormulaChunker"]


class FormulaChunker(BaseChunker):
    section_type = SectionType.FORMULA

    async def chunk(self, block: ContentBlock, processed: ProcessedBlock, *, kb_id: UUID, doc_type: DocumentType, chunking_version: str, start_index: int = 0) -> list[ChunkDraft]:
        formula = block.formula
        text = (formula.latex if formula else "") or processed.text or block.text
        if not text.strip():
            return []
        embedding = "\n".join(item for item in (text.strip(), processed.text) if item)
        return [self.build_chunk(block=block, processed=processed, kb_id=kb_id, index=start_index, text=text.strip(), embedding_text=embedding, section_type=SectionType.FORMULA)]
