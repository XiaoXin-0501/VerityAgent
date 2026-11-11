"""表格切片：小表整表；大表按行组切并重复表头。"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, QualityFlag, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock, TableBlock
from utils.tokens import estimate_tokens

from .base import BaseChunker

__all__ = ["TableChunker"]


class TableChunker(BaseChunker):
    section_type = SectionType.TABLE

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
        table = block.table
        if table is None:
            text = (processed.text or block.text).strip()
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
                    section_type=SectionType.TABLE,
                )
            ]

        header = self._row_text(table.headers)
        summary = table.summary or ""
        markdown = table.markdown or self._markdown(table)
        embedding_base = "\n".join(
            item for item in (table.caption or "", header, summary, markdown) if item
        )
        rows = table.rows
        if not rows:
            return [
                self.build_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=start_index,
                    text=markdown,
                    embedding_text=embedding_base,
                    section_type=SectionType.TABLE,
                    quality_flags=self._table_flags(table),
                )
            ]

        chunks: list[ChunkDraft] = []
        index = start_index
        group: list[list[str]] = []
        group_tokens = estimate_tokens(header)
        for row in rows:
            row_tokens = estimate_tokens(self._row_text(row))
            if group and group_tokens + row_tokens > self.max_tokens:
                text = "\n".join([header] + [self._row_text(item) for item in group])
                chunks.append(
                    self.build_chunk(
                        block=block,
                        processed=processed,
                        kb_id=kb_id,
                        index=index,
                        text=text,
                        embedding_text=f"{embedding_base}\n{text}",
                        section_type=SectionType.TABLE,
                        quality_flags=self._table_flags(table),
                    )
                )
                index += 1
                group = []
                group_tokens = estimate_tokens(header)
            group.append(row)
            group_tokens += row_tokens
        if group:
            text = "\n".join([header] + [self._row_text(item) for item in group])
            chunks.append(
                self.build_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=index,
                    text=text,
                    embedding_text=f"{embedding_base}\n{text}",
                    section_type=SectionType.TABLE,
                    quality_flags=self._table_flags(table),
                )
            )
        return chunks

    @staticmethod
    def _row_text(row: list[str]) -> str:
        return " | ".join(str(cell) for cell in row)

    def _markdown(self, table: TableBlock) -> str:
        lines = [self._row_text(table.headers)]
        lines.append(" | ".join("---" for _ in table.headers))
        lines.extend(self._row_text(row) for row in table.rows)
        return "\n".join(lines)

    @staticmethod
    def _table_flags(table: TableBlock) -> set[QualityFlag]:
        if table.confidence is not None and table.confidence < 0.6:
            return {QualityFlag.LOW_TABLE}
        return set()
