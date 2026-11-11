"""PPTX parser（python-pptx）：一页一页提取文本/备注/表格。"""

from __future__ import annotations

import io
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import make_block, make_parsed_document

__all__ = ["PptxParser"]


class PptxParser:
    parser_id = "pptx_python_pptx"
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
        try:
            from pptx import Presentation  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 python-pptx，无法解析 PPTX",
                details={"parser_id": self.parser_id},
            ) from exc
        presentation = Presentation(io.BytesIO(data))
        blocks = []
        order = 0
        for slide_index, slide in enumerate(presentation.slides, start=1):
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    text = shape.text.strip()
                    if text:
                        blocks.append(
                            make_block(
                                doc_id=doc_id,
                                page_no=slide_index,
                                order=order,
                                block_type=BlockType.PARAGRAPH,
                                text=text,
                                parser_id=self.parser_id,
                            )
                        )
                        order += 1
                if getattr(shape, "has_table", False):
                    table = shape.table
                    rows = [
                        [cell.text.strip() for cell in row.cells]
                        for row in table.rows
                    ]
                    if rows:
                        blocks.append(
                            make_block(
                                doc_id=doc_id,
                                page_no=slide_index,
                                order=order,
                                block_type=BlockType.TABLE,
                                text="\n".join(" | ".join(row) for row in rows),
                                parser_id=self.parser_id,
                                table=TableBlock(headers=rows[0], rows=rows[1:]),
                            )
                        )
                        order += 1
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=slide_index,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=notes,
                        parser_id=self.parser_id,
                        metadata={"notes": True},
                    )
                )
                order += 1
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.PPTX,
                    "has_text_layer": True,
                    "page_count": len(presentation.slides),
                    "text_coverage": 1.0,
                }
            ),
            blocks=blocks,
        )
