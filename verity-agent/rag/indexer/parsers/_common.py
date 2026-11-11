"""Parser 共用辅助：确定性 block id、文本解码、ParsedDocument 构造。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType
from rag.shared.ids import deterministic_block_id
from rag.shared.models import (
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
)

__all__ = ["decode_text", "make_block", "make_parsed_document"]


def decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def make_block(
    *,
    doc_id: UUID,
    page_no: int,
    order: int,
    block_type: BlockType,
    text: str,
    parser_id: str,
    **kwargs: Any,
) -> ContentBlock:
    return ContentBlock(
        block_id=deterministic_block_id(
            doc_id=doc_id,
            page_no=page_no,
            order=order,
            block_type=block_type.value,
        ),
        doc_id=doc_id,
        page_no=page_no,
        order=order,
        block_type=block_type,
        text=text,
        source_parser=parser_id,
        **kwargs,
    )


def make_parsed_document(
    *,
    source: DocumentSource,
    doc_id: UUID,
    kb_id: UUID,
    parser_id: str,
    parser_version: str,
    profile: DocumentProfile,
    blocks: list[ContentBlock],
    warnings: list[str] | None = None,
) -> ParsedDocument:
    return ParsedDocument(
        doc_id=doc_id,
        kb_id=kb_id,
        parser_id=parser_id,
        parser_version=parser_version,
        profile=profile,
        blocks=blocks,
        warnings=warnings or [],
    )
