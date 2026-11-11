"""代码切片：一个符号块一个 chunk，不拆函数体、不 overlap。"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import (
    CodeParseStatus,
    DocumentType,
    QualityFlag,
    SectionType,
)
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock

from .base import BaseChunker

__all__ = ["CodeChunker"]


class CodeChunker(BaseChunker):
    section_type = SectionType.CODE

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
        code = block.code
        body = (processed.text or (code.body if code else block.text) or "").strip()
        if not body:
            return []
        parts = [
            code.file_path if code else None,
            code.symbol_path if code else None,
            code.signature if code else None,
            code.docstring if code else None,
            body,
        ]
        embedding_text = "\n".join(item for item in parts if item)
        flags: set[QualityFlag] = set(processed.quality_flags)
        if code is not None and code.parse_status is CodeParseStatus.FALLBACK:
            flags.add(QualityFlag.CODE_PARSE_FALLBACK)
        return [
            self.build_chunk(
                block=block,
                processed=processed,
                kb_id=kb_id,
                index=start_index,
                text=body,
                embedding_text=embedding_text,
                section_type=SectionType.CODE,
                quality_flags=flags,
                metadata={
                    "language": code.language if code else None,
                    "symbol_path": code.symbol_path if code else None,
                    "start_line": code.start_line if code else None,
                    "end_line": code.end_line if code else None,
                },
            )
        ]
