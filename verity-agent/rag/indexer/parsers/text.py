"""纯文本 parser：按段落/空行切块。"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

from ._common import decode_text, make_block, make_parsed_document

__all__ = ["TextParser"]


class TextParser:
    parser_id = "text"
    parser_version = "1"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n").strip()
        paragraphs = [
            re.sub(r"[ \t]+", " ", item).strip()
            for item in re.split(r"\n\s*\n", text)
            if item.strip()
        ]
        blocks = [
            make_block(
                doc_id=doc_id,
                page_no=0,
                order=index,
                block_type=BlockType.PARAGRAPH,
                text=paragraph,
                parser_id=self.parser_id,
            )
            for index, paragraph in enumerate(paragraphs)
        ]
        has_text = bool(blocks)
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.TEXT,
                    "has_text_layer": has_text,
                    "page_count": 1,
                    "text_coverage": 1.0 if has_text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["文本文件没有可索引内容"],
        )
