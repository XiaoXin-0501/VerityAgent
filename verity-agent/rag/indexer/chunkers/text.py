"""文本/列表切片策略。

- ``TextChunker``：按句子/段落单元打包，默认目标 300、硬上限 600（由
  `IndexerConfig.chunk_size_tokens / max_chunk_tokens` 控制），overlap 只用于
  正文；中文单元之间不插空格（由 ``smart_join`` 处理）；
- ``ListChunker``：按列表项打包，保留换行与项目符号，不把列表压成一段话；
- 超长单元先按 token 切分（不丢内容），只有实际截断才标 ``TRUNCATED``。

打包逻辑与跨章节切片共用 ``packing.pack_units``，保证边界规则一致。
"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, QualityFlag, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from utils.text import smart_join, split_sentences

from .base import BaseChunker, section_type_for
from .packing import Unit, build_group_chunk, pack_units

__all__ = ["TextChunker", "ListChunker", "OcrChunker"]


def _units_for_block(
    block: ContentBlock,
    text: str,
    *,
    section_type: SectionType,
    quality_flags: set[QualityFlag],
) -> list[Unit]:
    """把一个块拆成最小切片单元，并附加来源信息。"""
    value = text.strip()
    if not value:
        return []
    if block.block_type in (BlockType.CODE,):
        # 代码按原样作为整体单元；超长时由 pack_units 按 token 切，不按句号拆。
        pieces = [value]
    elif section_type is SectionType.LIST:
        pieces = [line.strip() for line in value.split("\n") if line.strip()]
    else:
        pieces = split_sentences(value) or [value]
    return [
        Unit(
            text=piece,
            block_id=block.block_id,
            page_no=block.page_no or None,
            language=block.language,
            quality_flags=set(quality_flags),
        )
        for piece in pieces
        if piece
    ]


class TextChunker(BaseChunker):
    """正文切片：句子为最小单元，按 token 预算打包。"""

    section_type = SectionType.PARAGRAPH
    default_joiner = staticmethod(smart_join)

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
        """把正文块切成若干 chunk；空文本返回空列表。"""
        section_type = section_type_for(block.block_type)
        units = _units_for_block(
            block,
            processed.text or block.text or "",
            section_type=section_type,
            quality_flags=set(processed.quality_flags),
        )
        if not units:
            return []
        groups = pack_units(
            units,
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            overlap_tokens=self.overlap_tokens,
        )
        return [
            build_group_chunk(
                self,
                group,
                doc_id=block.doc_id,
                kb_id=kb_id,
                index=start_index + offset,
                block_type=block.block_type,
                section_type=section_type,
                heading_text=block.heading_text,
                heading_path=block.heading_path,
                joiner=self.default_joiner,
            )
            for offset, group in enumerate(groups)
        ]


class ListChunker(TextChunker):
    """列表切片：按列表项打包，保留换行与项目符号。"""

    section_type = SectionType.LIST
    default_joiner = staticmethod(lambda parts: "\n".join(parts))


class OcrChunker(TextChunker):
    """OCR 文本切片：沿用文本打包，低置信度由 quality_flags 标记。"""

    section_type = SectionType.PARAGRAPH
