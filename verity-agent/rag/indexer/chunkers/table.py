"""表格切片：小表整表；大表按行组切并重复表头。

规则（《RAG 模块数据结构设计》§5.2 / §5.4）：
- 小表（整表 token 不超过 ``max_tokens``）整表一个 chunk；
- 大表按 ``row_group_size`` 行或 token 上限分组，每组都重复表头；
- 表格不 overlap、不拆行；单行超长时保留原行，由质量闸门标记；
- 正文输出 Markdown 表格，embedding 额外带 caption/表头/模板摘要。
"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import (
    BlockType,
    DocumentType,
    QualityFlag,
    SectionType,
)
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock, TableBlock
from utils.text import smart_join, split_sentences, table_to_markdown, truncate
from utils.tokens import estimate_tokens, split_by_tokens

from .base import BaseChunker, semantic_embedding_text
from .describer import TableDescriber, TableDescriptionRequest
from .packing import Unit, pack_units

__all__ = ["TableChunker"]

# 表题/续表：要求“表/附表/续表 + 数字”或“Table + 数字”，避免把
# “表面积/表示”等正常列名误迁移为 caption。
_CAPTION_RE = re.compile(
    r"^\s*(?:(?:附?表|续表)\s*(?:[A-Z]?\d+|[一二三四五六七八九十]+)|table\s*[A-Z]?\d)",
    re.IGNORECASE,
)


def _is_caption_cell(cell: str) -> bool:
    """判断单元格是否像表题，而不是普通列名。"""
    return bool(cell) and bool(_CAPTION_RE.match(cell))


def _looks_like_prose_grid(
    headers: list[str],
    rows: list[list[str]],
) -> bool:
    """识别“页面网格把正文段落误判成表格”的结构。

    典型形态：列很多（>=4）、绝大多数单元格为空、每行基本只有一列长文本。
    真正的表格即使有合并单元格，也不会出现“整行只有一段长文、其余全空”的
    重复模式。命中时按正文切片，避免产生超长/无表头的伪表格 chunk。
    """
    if len(headers) < 4 or len(rows) < 3:
        return False
    width = len(headers)
    nonempty = sum(1 for cell in headers if cell)
    long_cells = 0
    rows_with_single_cell = 0
    for row in rows:
        values = [cell for cell in row if cell]
        nonempty += len(values)
        if any(len(cell) > 120 for cell in values):
            long_cells += 1
        if len(values) <= 1:
            rows_with_single_cell += 1
    total_cells = width * (len(rows) + 1)
    density = nonempty / max(1, total_cells)
    return (
        density < 0.25
        and long_cells >= 1
        and rows_with_single_cell / len(rows) >= 0.6
    )


def _flatten_grid(headers: list[str], rows: list[list[str]]) -> str:
    """把伪表格按阅读顺序压成正文行，保留非空单元格内容。"""
    parts: list[str] = [cell for cell in headers if cell]
    for row in rows:
        parts.extend(cell for cell in row if cell)
    return "\n".join(dict.fromkeys(parts)).strip()


def _split_overlong_rows(
    headers: list[str],
    rows: list[list[str]],
    *,
    max_tokens: int,
    overhead_tokens: int,
) -> list[list[str]]:
    """把“单个单元格占绝大部分内容”的超长行拆成多行。

    常见于 PDF 把合并单元格/段落塞进一行；直接作为一个 chunk 会超过
    max_tokens。只在长单元格占该行内容 70% 以上时拆分，其它列作为重复
    上下文；多列都超长时保留原行，由质量闸门标记，避免把表格语义拆碎。
    """
    # 留 10% 余量：拆分后的片段仍要和其它列/表头/分隔线一起计算 token。
    budget = max(50, int((max_tokens - overhead_tokens) * 0.9))
    expanded: list[list[str]] = []
    for row in rows:
        if (
            estimate_tokens(table_to_markdown(headers, [row]))
            + overhead_tokens
            <= max_tokens
        ):
            expanded.append(row)
            continue
        lengths = [len(cell) for cell in row]
        total = sum(lengths)
        if total <= 0:
            expanded.append(row)
            continue
        index = max(range(len(row)), key=lambda item: lengths[item])
        if lengths[index] / total < 0.7:
            expanded.append(row)
            continue
        pieces = split_by_tokens(row[index], budget)
        if len(pieces) <= 1:
            expanded.append(row)
            continue
        for piece in pieces:
            split_row = list(row)
            split_row[index] = piece.strip()
            expanded.append(split_row)
    return expanded


def _looks_like_header_row(row: list[str]) -> bool:
    """判断一行是否像表头（短标签、非纯数值、非长句）。

    PDF 多级表头的子列名通常很短；数据行常包含数值、长文本或空列。
    这个判断只用于“稀疏表头合并”，宁可少合并也不能吞掉数据行。
    """
    cells = [cell for cell in row if cell]
    if len(cells) < 2:
        return False
    if any(len(cell) > 60 for cell in cells):
        return False
    numeric = sum(
        1
        for cell in cells
        if cell.replace(".", "", 1).replace("%", "", 1).replace("-", "", 1).isdigit()
    )
    return numeric / len(cells) < 0.7


def _normalize_table(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]], str | None]:
    """修正 PDF 表格常见的“表题/多级表头被当成表头”结构。

    PDF 解析器常出现两类结构问题：

    1. ``表 1. xxx | | |`` 被识别为表头，真正的列名在第一行数据；
    2. 前两列是行标题（作者/项目），其余列是分组表头，第二行才是子列名。

    这里用保守规则修复：把像“表 X”的单元格迁移为 caption；当表头非空列
    少于一半、且下一行像短表头标签时，把该行合并进表头并删除。最多合并
    两层表头，避免把数据行当表头。普通表格（表头有多个真实列名、第一行
    是数据）不会被改动。返回 ``(headers, rows, promoted_caption)``。
    """
    width = max([len(headers), *(len(row) for row in rows)], default=0)
    if width == 0:
        return [], [], None
    current_headers = [
        "" if index >= len(headers) else str(headers[index] or "").strip()
        for index in range(width)
    ]
    current_rows = [
        [
            "" if index >= len(row) else str(row[index] or "").strip()
            for index in range(width)
        ]
        for row in rows
    ]
    # 去掉全空数据行；空行会把表题/表头推到错误位置。
    current_rows = [
        row for row in current_rows if any(cell for cell in row)
    ]

    caption: str | None = None
    for index, cell in enumerate(current_headers):
        if _is_caption_cell(cell):
            # “表 1./续表”是 caption，不是列名；清空后让真实表头补位。
            if caption is None:
                caption = cell
            current_headers[index] = ""

    # 最多合并两层表头：第一层可能是“作者/项目”行标题，第二层才是分组子列名。
    for _ in range(2):
        if len(current_headers) < 3 or not current_rows:
            break
        nonempty = sum(1 for cell in current_headers if cell)
        if nonempty >= max(2, (len(current_headers) + 1) // 2):
            break
        first = current_rows[0]
        if not _looks_like_header_row(first):
            break
        if sum(1 for cell in first if cell) <= nonempty:
            break
        # 表头原本有值的列若在本行也有值，说明本行可能混有数据，不能删。
        if any(
            current_headers[index] and first[index]
            for index in range(len(current_headers))
        ):
            break
        merged = list(current_headers)
        for index, cell in enumerate(first):
            if not merged[index] and cell:
                merged[index] = cell
        current_headers = merged
        current_rows = current_rows[1:]

    return current_headers, current_rows, caption


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
        headers, rows, promoted_caption = _normalize_table(headers, rows)
        if not caption and promoted_caption:
            # 原表头里的“表 1.”文本迁移为 caption，正文与向量都保留表题。
            caption = promoted_caption
        context_tokens = estimate_tokens(context)
        caption_tokens = estimate_tokens(caption)
        rows = _split_overlong_rows(
            headers,
            rows,
            max_tokens=self.max_tokens,
            overhead_tokens=(
                estimate_tokens(table_to_markdown(headers, []))
                + context_tokens
                + caption_tokens
            ),
        )
        flags = self._table_flags(table) | set(processed.quality_flags)
        # 页面网格把正文段落误判成表格时，按正文切片并保留上下文；否则会
        # 产出没有表头、单行 800+ token 的伪表格 chunk。
        if _looks_like_prose_grid(headers, rows):
            return self._build_prose_chunks(
                block=block,
                processed=processed,
                kb_id=kb_id,
                index=start_index,
                text=_flatten_grid(headers, rows),
                context=context,
                context_ids=context_ids,
                flags=flags,
            )
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
            and estimate_tokens(whole) + context_tokens + caption_tokens
            <= self.max_tokens
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
        group_tokens = (
            estimate_tokens(table_to_markdown(headers, []))
            + context_tokens
            + caption_tokens
        )
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
                estimate_tokens(table_to_markdown(headers, []))
                + context_tokens
                + caption_tokens
            )

        for offset, row in enumerate(rows, start=1):
            # 用真实 Markdown 重新估算，而不是把“ | ”拼接后的行 token 相加：
            # 表头/分隔线/单元格对齐开销会让组超过 max_tokens。
            candidate_tokens = (
                estimate_tokens(table_to_markdown(headers, [*group, row]))
                + context_tokens
                + caption_tokens
            )
            if group and (
                len(group) >= self.row_group_size
                or candidate_tokens > self.max_tokens
            ):
                emit()
                group_start = offset
            elif not group:
                group_start = offset
            group.append(row)
            group_tokens = candidate_tokens

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

    def _build_prose_chunks(
        self,
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        index: int,
        text: str,
        context: str,
        context_ids: list[UUID],
        flags: set[QualityFlag],
    ) -> list[ChunkDraft]:
        """把伪表格正文按普通段落打包，保留标题与吸附上下文。"""
        body = f"{context}\n{text}".strip() if context else text.strip()
        if not body:
            return []
        units = [
            Unit(text=piece)
            for piece in (split_sentences(body) or [body])
            if piece.strip()
        ]
        groups = pack_units(
            units,
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        chunks: list[ChunkDraft] = []
        for group in groups:
            chunk_text = smart_join([unit.text for unit in group])
            embedding_text = (
                f"{block.heading_text}\n{chunk_text}"
                if block.heading_text
                else chunk_text
            )
            chunks.append(
                self.build_chunk_from_text(
                    doc_id=block.doc_id,
                    kb_id=kb_id,
                    block_type=BlockType.PARAGRAPH,
                    section_type=SectionType.PARAGRAPH,
                    index=index + len(chunks),
                    text=chunk_text,
                    embedding_text=embedding_text,
                    block_ids=[block.block_id, *context_ids],
                    page_no=block.page_no or None,
                    heading=block.heading_text,
                    heading_path=block.heading_path,
                    language=block.language,
                    quality_flags=set(flags),
                    metadata={"table_as_text": True},
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

        describer = self.table_describer
        need_llm = describer is not None and (
            is_spreadsheet or (not local and heading_short)
        )
        if not need_llm:
            return local
        assert describer is not None

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
        generated = await describer.describe(request)
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
        # 旧策略：heading + 局部描述 + 表头 + 摘要 + 整表。
        # 仅在新策略无可用 heading/描述时兜底，避免产出空向量。
        # 文档描述不进入 embedding（D-103），只保留在 metadata。
        legacy_parts: list[str] = []
        for item in (
            block.heading_text,
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
        # 只有拿到“语义描述”时才用 heading+描述向量；没有描述时（默认表格
        # 摘要 LLM 关闭、普通 PDF 无 caption）必须回退到 legacy 全文，否则
        # 表头/表体完全不进向量，表格召回会退化成只按 heading 猜。
        if (
            self.semantic_embedding_enabled
            and semantic_description.strip()
            and semantic_text
        ):
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
