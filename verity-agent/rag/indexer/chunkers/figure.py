"""图注切片：caption + OCR + 邻近文本。

v1 不注册该策略；文件保留给后续图片/图注接入。
"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import DocumentType, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock

from .base import BaseChunker

__all__ = ["FigureChunker"]


class FigureChunker(BaseChunker):
    """把图注/OCR 文本切成独立 chunk。"""

    section_type = SectionType.FIGURE

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
        """按 caption + OCR + 邻近文本组装 chunk；无文本返回空列表。"""
        figure = block.figure
        parts = [
            figure.caption if figure else None,
            figure.ocr_text if figure else None,
            processed.text or block.text,
        ]
        text = "\n".join(item for item in parts if item).strip()
        if not text:
            return []
        return [
            self.build_chunk(
                block=block,
                processed=processed,
                kb_id=kb_id,
                index=start_index,
                text=text,
                embedding_text=text,
                section_type=SectionType.FIGURE,
                metadata={
                    "image_ref": figure.image_ref if figure else None
                },
            )
        ]
