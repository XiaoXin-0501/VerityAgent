"""文本/列表切片策略。

按句子/段落单元打包，target/max token 由 IndexerConfig 控制；overlap 只用于
正文和列表，表格/代码/公式不 overlap。
"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import DocumentType, SectionType
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from utils.tokens import estimate_tokens, split_by_tokens

from .base import BaseChunker

__all__ = ["TextChunker", "ListChunker", "OcrChunker"]

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s*|\n+")


class TextChunker(BaseChunker):
    section_type = SectionType.PARAGRAPH

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
        text = (processed.text or block.text or "").strip()
        if not text:
            return []
        units = [item.strip() for item in _SENTENCE_SPLIT.split(text) if item.strip()]
        if not units:
            units = [text]
        chunks = self._pack(
            units,
            block=block,
            processed=processed,
            kb_id=kb_id,
            start_index=start_index,
        )
        return chunks

    def _pack(
        self,
        units: list[str],
        *,
        block: ContentBlock,
        processed: ProcessedBlock,
        kb_id: UUID,
        start_index: int,
    ) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        current: list[str] = []
        current_tokens = 0
        index = start_index
        for unit in units:
            unit_tokens = estimate_tokens(unit)
            if current and current_tokens + unit_tokens > self.max_tokens:
                text = " ".join(current).strip()
                chunks.append(
                    self.build_chunk(
                        block=block,
                        processed=processed,
                        kb_id=kb_id,
                        index=index,
                        text=text,
                        embedding_text=self.embedding_text(block, text),
                        section_type=self.section_type,
                    )
                )
                index += 1
                current = self._tail(current)
                current_tokens = estimate_tokens(" ".join(current))
            if unit_tokens > self.max_tokens:
                # 超长单元按 token 上限切分；这是切分，不是截断
                for piece in split_by_tokens(unit, self.max_tokens):
                    if current:
                        text = " ".join(current).strip()
                        chunks.append(
                            self.build_chunk(
                                block=block,
                                processed=processed,
                                kb_id=kb_id,
                                index=index,
                                text=text,
                                embedding_text=self.embedding_text(block, text),
                                section_type=self.section_type,
                            )
                        )
                        index += 1
                        current = []
                        current_tokens = 0
                    text = piece
                    chunks.append(
                        self.build_chunk(
                            block=block,
                            processed=processed,
                            kb_id=kb_id,
                            index=index,
                            text=text,
                            embedding_text=self.embedding_text(block, text),
                            section_type=self.section_type,
                        )
                    )
                    index += 1
                continue
            current.append(unit)
            current_tokens += unit_tokens
            if current_tokens >= self.target_tokens:
                text = " ".join(current).strip()
                chunks.append(
                    self.build_chunk(
                        block=block,
                        processed=processed,
                        kb_id=kb_id,
                        index=index,
                        text=text,
                        embedding_text=self.embedding_text(block, text),
                        section_type=self.section_type,
                    )
                )
                index += 1
                current = self._tail(current)
                current_tokens = estimate_tokens(" ".join(current))
        if current:
            text = " ".join(current).strip()
            if text:
                chunks.append(
                    self.build_chunk(
                        block=block,
                        processed=processed,
                        kb_id=kb_id,
                        index=index,
                        text=text,
                        embedding_text=self.embedding_text(block, text),
                        section_type=self.section_type,
                    )
                )
        return chunks

    def _tail(self, units: list[str]) -> list[str]:
        if self.overlap_tokens <= 0:
            return []
        tail: list[str] = []
        tokens = 0
        for unit in reversed(units):
            unit_tokens = estimate_tokens(unit)
            if tokens + unit_tokens > self.overlap_tokens:
                break
            tail.insert(0, unit)
            tokens += unit_tokens
        return tail


class ListChunker(TextChunker):
    section_type = SectionType.LIST


class OcrChunker(TextChunker):
    section_type = SectionType.PARAGRAPH
