"""代码切片：按行/逻辑块打包，保留签名、docstring 和相邻描述。

v1 不做 tree-sitter AST 切分（Markdown 围栏代码按文本处理），但会：
- 不跨代码块边界；
- 单块超长时按行/ token 切分，每块重复文件路径、符号、签名和描述；
- 紧贴代码的短描述（由关联阶段提供）写入 chunk 正文与 embedding，
  避免代码脱离语义。
"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import (
    CodeParseStatus,
    DocumentType,
    QualityFlag,
    SectionType,
)
from rag.shared.models import ChunkDraft, ContentBlock, ProcessedBlock
from utils.tokens import estimate_tokens, split_by_tokens

from .base import BaseChunker
from .packing import Unit, pack_units

__all__ = ["CodeChunker"]


class CodeChunker(BaseChunker):
    """代码块切片器。"""

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
        """把代码块切成一个或多个 chunk；不跨代码块、不做 overlap。"""
        code = block.code
        body = (
            processed.text
            or (code.body if code else "")
            or block.text
            or ""
        ).strip()
        if not body:
            return []

        context = (processed.metadata.get("context_text") or "").strip()
        context_ids = list(processed.metadata.get("context_block_ids") or [])
        document_description = (
            processed.metadata.get("document_description") or ""
        ).strip()
        flags: set[QualityFlag] = set(processed.quality_flags)
        if code is not None and code.parse_status is CodeParseStatus.FALLBACK:
            flags.add(QualityFlag.CODE_PARSE_FALLBACK)

        # 描述也占 token；先为描述预留预算，保证代码 chunk 不超过 max_tokens。
        context_tokens = estimate_tokens(context)
        body_max = max(1, self.max_tokens - context_tokens)
        body_target = max(1, min(self.target_tokens, body_max))
        units = self._units_for_code(body, body_max)
        groups = pack_units(
            units,
            target_tokens=body_target,
            max_tokens=body_max,
            overlap_tokens=0,
        )
        if not groups:
            return []

        prefix_parts = [
            block.heading_text,
            document_description,
            code.file_path if code else None,
            code.symbol_path if code else None,
            code.signature if code else None,
            code.docstring if code else None,
            context,
        ]
        prefix = "\n".join(
            dict.fromkeys(item for item in prefix_parts if item)
        )
        chunks: list[ChunkDraft] = []
        for offset, group in enumerate(groups):
            code_text = "\n".join(unit.text for unit in group).strip()
            if not code_text:
                continue
            text = f"{context}\n{code_text}".strip() if context else code_text
            embedding_text = f"{prefix}\n{code_text}".strip()
            chunks.append(
                self.build_chunk_from_text(
                    doc_id=block.doc_id,
                    kb_id=kb_id,
                    block_type=block.block_type,
                    section_type=SectionType.CODE,
                    index=start_index + len(chunks),
                    text=text,
                    embedding_text=embedding_text,
                    block_ids=[block.block_id, *context_ids],
                    page_no=block.page_no or None,
                    heading=block.heading_text,
                    heading_path=block.heading_path,
                    language=block.language,
                    quality_flags=flags,
                    metadata={
                        "language": code.language if code else None,
                        "symbol_path": code.symbol_path if code else None,
                        "start_line": code.start_line if code else None,
                        "end_line": code.end_line if code else None,
                        "context": context or None,
                        "context_block_ids": [str(item) for item in context_ids],
                        "document_description": document_description or None,
                    },
                )
            )
        return chunks

    @staticmethod
    def _units_for_code(body: str, max_tokens: int) -> list[Unit]:
        """按行拆分代码；单行超长时再按 token 切分。"""
        units: list[Unit] = []
        for line in body.split("\n"):
            if estimate_tokens(line) <= max_tokens:
                units.append(Unit(text=line))
                continue
            for piece in split_by_tokens(line, max_tokens):
                if piece.strip():
                    units.append(Unit(text=piece))
        return units
