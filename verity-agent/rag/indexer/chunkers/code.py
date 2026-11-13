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
from rag.shared.models import (
    ChunkDraft,
    CodeBlock,
    ContentBlock,
    ProcessedBlock,
)
from utils.text import truncate
from utils.tokens import estimate_tokens, split_by_tokens

from .base import BaseChunker, semantic_embedding_text
from .describer import CodeDescriber, CodeDescriptionRequest
from .packing import Unit, pack_units

__all__ = ["CodeChunker"]


class CodeChunker(BaseChunker):
    """代码块切片器。"""

    section_type = SectionType.CODE

    def __init__(
        self,
        *,
        code_describer: CodeDescriber | None = None,
        semantic_embedding_enabled: bool = True,
        semantic_heading_min_tokens: int = 4,
        semantic_description_max_chars: int = 200,
        semantic_code_max_input_chars: int = 2000,
        **kwargs,
    ) -> None:
        """保存语义描述器与向量化策略。"""
        super().__init__(**kwargs)
        self.code_describer = code_describer
        self.semantic_embedding_enabled = semantic_embedding_enabled
        self.semantic_heading_min_tokens = semantic_heading_min_tokens
        self.semantic_description_max_chars = semantic_description_max_chars
        self.semantic_code_max_input_chars = semantic_code_max_input_chars

    async def _semantic_description(
        self,
        *,
        block: ContentBlock,
        code: CodeBlock | None,
        context: str,
        body: str,
    ) -> str:
        """决定代码 chunk 的描述：context/docstring 优先，必要时 LLM 兜底。"""
        docstring = (code.docstring if code else "") or ""
        local = "\n".join(
            dict.fromkeys(
                item.strip()
                for item in (context, docstring)
                if item and item.strip()
            )
        )
        heading = block.heading_text or ""
        heading_short = (
            estimate_tokens(heading) < self.semantic_heading_min_tokens
        )
        if (
            self.code_describer is None
            or local
            or not heading_short
        ):
            return local

        request = CodeDescriptionRequest(
            heading=heading,
            file_path=(code.file_path if code else "") or "",
            symbol_path=(code.symbol_path if code else "") or "",
            signature=(code.signature if code else "") or "",
            docstring=docstring,
            context=context,
            code=truncate(body, self.semantic_code_max_input_chars),
        )
        generated = await self.code_describer.describe(request)
        if generated and generated.strip():
            return truncate(
                generated.strip(),
                self.semantic_description_max_chars,
            )
        return local

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
        semantic_description = await self._semantic_description(
            block=block,
            code=code,
            context=context,
            body=body,
        )

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
        semantic_text = semantic_embedding_text(
            block.heading_text,
            semantic_description,
        )
        chunks: list[ChunkDraft] = []
        for offset, group in enumerate(groups):
            code_text = "\n".join(unit.text for unit in group).strip()
            if not code_text:
                continue
            text = f"{context}\n{code_text}".strip() if context else code_text
            if self.semantic_embedding_enabled and semantic_text:
                embedding_text = semantic_text
            else:
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
                    semantic_description=semantic_description or None,
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
