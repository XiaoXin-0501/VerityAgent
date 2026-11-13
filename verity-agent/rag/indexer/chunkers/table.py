"""表格切片：小表整表；大表按行组切并重复表头。

规则（《RAG 模块数据结构设计》§5.2 / §5.4）：
- 小表（整表 token 不超过 ``max_tokens``）整表一个 chunk；
- 大表按 ``row_group_size`` 行或 token 上限分组，每组都重复表头；
- 表格不 overlap、不拆行；单行超长时保留原行，由质量闸门标记；
- 正文输出 Markdown 表格，embedding 额外带 caption/表头/模板摘要。
"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import DocumentType, QualityFlag, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock, TableBlock
from utils.text import table_to_markdown
from utils.tokens import estimate_tokens

from .base import BaseChunker

__all__ = ["TableChunker"]

# 表格识别置信度低于该值时标记 LOW_TABLE；与 QualityPolicy 默认值一致。
_MIN_TABLE_CONFIDENCE = 0.6


class TableChunker(BaseChunker):
    """表格/工作表切片器。"""

    section_type = SectionType.TABLE

    def __init__(self, *, row_group_size: int = 20, **kwargs) -> None:
        """``row_group_size`` 控制大表每个 chunk 的数据行数。"""
        super().__init__(**kwargs)
        # 大表每个 chunk 最多包含多少数据行（不含重复表头）。
        self.row_group_size = max(1, row_group_size)

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
        """把表格块切成一个或多个 chunk。"""
        context = (processed.metadata.get("context_text") or "").strip()
        context_ids = list(processed.metadata.get("context_block_ids") or [])
        document_description = (
            processed.metadata.get("document_description") or ""
        ).strip()
        table = block.table
        if table is None:
            # 没有结构化表格时退化为普通文本，避免丢内容。
            text = (processed.text or block.text or "").strip()
            if not text:
                return []
            content = f"{context}\n{text}".strip() if context else text
            embedding_parts = [
                item
                for item in (document_description, content)
                if item
            ]
            return [
                self.build_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=start_index,
                    text=content,
                    embedding_text="\n".join(embedding_parts),
                    section_type=SectionType.TABLE,
                    extra_block_ids=context_ids,
                    metadata=(
                        {"document_description": document_description}
                        if document_description
                        else {}
                    ),
                )
            ]

        caption = (table.caption or "").strip()
        summary = (table.summary or "").strip()
        headers = list(table.headers)
        rows = [list(row) for row in table.rows]
        flags = self._table_flags(table) | set(processed.quality_flags)
        context_tokens = estimate_tokens(context)

        # 整表（含表头、描述）在 token 上限内：一个 chunk 装下，避免过度切分。
        whole = table_to_markdown(headers, rows)
        if (
            whole
            and estimate_tokens(whole) + context_tokens <= self.max_tokens
        ):
            return [
                self._build_table_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=start_index,
                    caption=caption,
                    context=context,
                    context_ids=context_ids,
                    headers=headers,
                    summary=summary,
                    markdown=whole,
                    flags=flags,
                    row_start=1 if rows else 0,
                    row_end=len(rows),
                    row_total=len(rows),
                )
            ]

        chunks: list[ChunkDraft] = []
        group: list[list[str]] = []
        group_tokens = estimate_tokens(table_to_markdown(headers, [])) + context_tokens
        group_start = 1
        index = start_index

        def emit() -> None:
            """输出当前行组，并重复表头与描述。"""
            nonlocal group, group_tokens, index, group_start
            if not group:
                return
            markdown = table_to_markdown(headers, group)
            chunks.append(
                self._build_table_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=index,
                    caption=caption,
                    context=context,
                    context_ids=context_ids,
                    headers=headers,
                    summary=summary,
                    markdown=markdown,
                    flags=flags,
                    row_start=group_start,
                    row_end=group_start + len(group) - 1,
                    row_total=len(rows),
                )
            )
            index += 1
            group = []
            group_tokens = (
                estimate_tokens(table_to_markdown(headers, [])) + context_tokens
            )

        for offset, row in enumerate(rows, start=1):
            row_tokens = estimate_tokens(" | ".join(str(cell) for cell in row))
            if group and (
                len(group) >= self.row_group_size
                or group_tokens + row_tokens > self.max_tokens
            ):
                emit()
                group_start = offset
            elif not group:
                group_start = offset
            group.append(row)
            group_tokens += row_tokens

        emit()

        # 只有表头、没有数据行：仍然输出一个 chunk，方便检索列名。
        if not chunks:
            chunks.append(
                self._build_table_chunk(
                    block=block,
                    processed=processed,
                    kb_id=kb_id,
                    index=index,
                    caption=caption,
                    context=context,
                    context_ids=context_ids,
                    headers=headers,
                    summary=summary,
                    markdown=table_to_markdown(headers, []),
                    flags=flags,
                    row_start=0,
                    row_end=0,
                    row_total=0,
                )
            )
        return chunks

    def _build_table_chunk(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        index: int,
        caption: str,
        context: str,
        context_ids: list[UUID],
        headers: list[str],
        summary: str,
        markdown: str,
        flags: set[QualityFlag],
        row_start: int,
        row_end: int,
        row_total: int,
    ) -> ChunkDraft:
        """组装表格 chunk：正文带描述 + Markdown，embedding 带完整上下文。"""
        document_description = (
            processed.metadata.get("document_description") or ""
        ).strip()
        header_text = " | ".join(str(cell) for cell in headers)
        text = "\n".join(
            item for item in (context, caption, markdown) if item
        ).strip()
        embedding_parts: list[str] = []
        for item in (
            block.heading_text,
            document_description,
            context,
            caption,
            header_text,
            summary,
            markdown,
        ):
            if item and item not in embedding_parts:
                embedding_parts.append(item)
        embedding_text = "\n".join(embedding_parts)
        metadata = {
            "caption": caption or None,
            "context": context or None,
            "context_block_ids": [str(item) for item in context_ids],
            "document_description": document_description or None,
            "headers": headers,
            "summary": summary or None,
            "row_start": row_start,
            "row_end": row_end,
            "row_total": row_total,
            "sheet": block.metadata.get("sheet"),
        }
        return self.build_chunk(
            block=block,
            processed=processed,
            kb_id=kb_id,
            index=index,
            text=text,
            embedding_text=embedding_text or text,
            section_type=SectionType.TABLE,
            quality_flags=flags,
            metadata=metadata,
            extra_block_ids=context_ids,
        )

    @staticmethod
    def _table_flags(table: TableBlock) -> set[QualityFlag]:
        """表格识别置信度过低时标记 ``LOW_TABLE``。"""
        if (
            table.confidence is not None
            and table.confidence < _MIN_TABLE_CONFIDENCE
        ):
            return {QualityFlag.LOW_TABLE}
        return set()
