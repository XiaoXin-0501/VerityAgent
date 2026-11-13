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
from utils.text import table_to_markdown, truncate
from utils.tokens import estimate_tokens

from .base import BaseChunker, semantic_embedding_text
from .describer import TableDescriber, TableDescriptionRequest

__all__ = ["TableChunker"]


class TableChunker(BaseChunker):
    """表格/工作表切片器。"""

    section_type = SectionType.TABLE

    def __init__(
        self,
        *,
        row_group_size: int = 20,
        min_table_confidence: float = 0.6,
        table_describer: TableDescriber | None = None,
        semantic_embedding_enabled: bool = True,
        semantic_heading_min_tokens: int = 4,
        semantic_preview_rows: int = 3,
        semantic_description_max_chars: int = 200,
        **kwargs,
    ) -> None:
        """``row_group_size`` 控制大表每个 chunk 的数据行数。

        ``min_table_confidence`` 来自 ``IndexerConfig.quality``，不在这里写死，
        保证与 QualityGate / SectionAwareChunker 使用同一阈值。

        ``table_describer`` 为空表示不走 LLM；语义向量化仍会使用已有
        context/caption/summary 作为描述，正文不进入向量。
        """
        super().__init__(**kwargs)
        # 大表每个 chunk 最多包含多少数据行（不含重复表头）。
        self.row_group_size = max(1, row_group_size)
        self.min_table_confidence = min_table_confidence
        self.table_describer = table_describer
        self.semantic_embedding_enabled = semantic_embedding_enabled
        self.semantic_heading_min_tokens = semantic_heading_min_tokens
        self.semantic_preview_rows = max(1, semantic_preview_rows)
        self.semantic_description_max_chars = (
            semantic_description_max_chars
        )

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
                    semantic_description=context or None,
                )
            ]

        caption = (table.caption or "").strip()
        summary = (table.summary or "").strip()
        headers = list(table.headers)
        rows = [list(row) for row in table.rows]
        flags = self._table_flags(table) | set(processed.quality_flags)
        context_tokens = estimate_tokens(context)
        semantic_description = await self._semantic_description(
            block=block,
            processed=processed,
            doc_type=doc_type,
            caption=caption,
            summary=summary,
            context=context,
            headers=headers,
            rows=rows,
        )

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
                    semantic_description=semantic_description,
                    is_spreadsheet=(
                        doc_type is DocumentType.SPREADSHEET
                    ),
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
                    semantic_description=semantic_description,
                    is_spreadsheet=(
                        doc_type is DocumentType.SPREADSHEET
                    ),
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
                    semantic_description=semantic_description,
                    is_spreadsheet=(
                        doc_type is DocumentType.SPREADSHEET
                    ),
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

    async def _semantic_description(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        doc_type: DocumentType,
        caption: str,
        summary: str,
        context: str,
        headers: list[str],
        rows: list[list[str]],
    ) -> str:
        """决定表格 chunk 的“描述”：本地上下文优先，LLM 生成语义描述。

        - 普通表格本地描述 = 相邻正文 + caption + summary；
        - Excel 的 caption 通常只是“文件名 / sheet”，不算语义描述；只使用
          相邻正文/工作簿描述作为本地描述；
        - Excel 启用 LLM 时总是生成描述；普通表格仅在“无本地描述且 heading
          过短”时调用 LLM；
        - LLM 未注入/失败时不伪造“表头+样例”式描述，返回本地描述（可能为空），
          由向量化侧回退旧策略。
        """
        context = context.strip()
        summary = summary.strip()
        caption = caption.strip()
        heading = block.heading_text or ""
        filename = str(block.metadata.get("filename") or "")
        sheet = str(block.metadata.get("sheet") or "")
        is_spreadsheet = doc_type is DocumentType.SPREADSHEET
        heading_short = (
            estimate_tokens(heading) < self.semantic_heading_min_tokens
        )

        local_parts = [context, summary]
        if not is_spreadsheet:
            local_parts.insert(1, caption)
        local = "\n".join(
            dict.fromkeys(item for item in local_parts if item)
        )

        need_llm = self.table_describer is not None and (
            is_spreadsheet or (not local and heading_short)
        )
        if not need_llm:
            return local

        request = TableDescriptionRequest(
            heading=heading,
            caption=caption,
            filename=filename,
            sheet=sheet,
            doc_type=doc_type.value,
            headers=headers,
            rows=[
                list(row)
                for row in rows[: self.semantic_preview_rows]
            ],
            context=context,
        )
        generated = await self.table_describer.describe(request)
        if generated and generated.strip():
            return truncate(
                generated.strip(),
                self.semantic_description_max_chars,
            )
        return local

    def _build_table_chunk(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        index: int,
        caption: str,
        context: str,
        semantic_description: str,
        is_spreadsheet: bool,
        context_ids: list[UUID],
        headers: list[str],
        summary: str,
        markdown: str,
        flags: set[QualityFlag],
        row_start: int,
        row_end: int,
        row_total: int,
    ) -> ChunkDraft:
        """组装表格 chunk：正文带描述 + Markdown，embedding 只带 heading+描述。"""
        document_description = (
            processed.metadata.get("document_description") or ""
        ).strip()
        header_text = " | ".join(str(cell) for cell in headers)
        text = "\n".join(
            item for item in (context, caption, markdown) if item
        ).strip()
        # 旧策略：heading + 文档描述 + 局部描述 + 表头 + 摘要 + 整表。
        # 仅在新策略无可用 heading/描述时兜底，避免产出空向量。
        legacy_parts: list[str] = []
        for item in (
            block.heading_text,
            document_description,
            context,
            caption,
            header_text,
            summary,
            markdown,
        ):
            if item and item not in legacy_parts:
                legacy_parts.append(item)
        legacy_embedding = "\n".join(legacy_parts)
        semantic_text = semantic_embedding_text(
            block.heading_text,
            semantic_description,
        )
        if (
            self.semantic_embedding_enabled
            and is_spreadsheet
            and not semantic_description
        ):
            # Excel 的 heading 通常只是文件名/sheet；没有语义描述时不要用
            # 这类 heading 单独生成向量，回退到包含表格正文的旧策略。
            embedding_text = legacy_embedding
        elif self.semantic_embedding_enabled and semantic_text:
            embedding_text = semantic_text
        else:
            embedding_text = legacy_embedding
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
            semantic_description=semantic_description or None,
        )

    def _table_flags(self, table: TableBlock) -> set[QualityFlag]:
        """表格识别置信度过低时标记 ``LOW_TABLE``。"""
        if (
            table.confidence is not None
            and table.confidence < self.min_table_confidence
        ):
            return {QualityFlag.LOW_TABLE}
        return set()
