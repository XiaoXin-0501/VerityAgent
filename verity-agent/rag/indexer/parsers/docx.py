"""DOCX parser（python-docx，v1 只处理文本与表格，不处理图片）。

- 按文档 body 顺序遍历段落和顶层表格，避免“所有段落后再所有表格”的错序；
- 段落只取 `paragraph.text`，内嵌图片不会进入 chunk；
- 表格转成 `TableBlock`，`text` 保留 ``a | b`` 的纯文本形式供切片/BM25；
- 页数无法从 DOCX 可靠获取，统一记为 1 页。
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import make_block, make_parsed_document

__all__ = ["DocxParser"]


def _iter_body_items(document) -> Iterator[object]:
    """按 body 顺序产出 Paragraph / Table。"""
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


class DocxParser:
    parser_id = "docx_python_docx"
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
            from docx import Document  # type: ignore[import-not-found]
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise RagError(
                RagErrorCode.INVALID_CONFIG,
                "未安装 python-docx，无法解析 DOCX",
                details={"parser_id": self.parser_id},
            ) from exc

        try:
            document = Document(io.BytesIO(data))
        except Exception as exc:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "DOCX 无法打开",
                details={"filename": source.filename, "cause": str(exc)},
            ) from exc

        blocks: list[ContentBlock] = []
        order = 0
        heading_text: str | None = None
        has_tables = False

        for item in _iter_body_items(document):
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style = (item.style.name or "") if item.style else ""
                if style.lower().startswith("heading"):
                    try:
                        level = int(style.split()[-1])
                    except ValueError:
                        level = 1
                    heading_text = text
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.HEADING,
                            text=text,
                            parser_id=self.parser_id,
                            heading_level=level,
                            heading_text=text,
                        )
                    )
                else:
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.PARAGRAPH,
                            text=text,
                            parser_id=self.parser_id,
                            heading_text=heading_text,
                        )
                    )
                order += 1
                continue

            if not isinstance(item, Table):
                continue
            rows = [
                [cell.text.strip() for cell in row.cells]
                for row in item.rows
            ]
            rows = [row for row in rows if any(cell for cell in row)]
            if not rows:
                continue
            has_tables = True
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=order,
                    block_type=BlockType.TABLE,
                    text="\n".join(" | ".join(row) for row in rows),
                    parser_id=self.parser_id,
                    table=TableBlock(headers=rows[0], rows=rows[1:]),
                    heading_text=heading_text,
                )
            )
            order += 1

        has_text = bool(blocks)
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.DOCX,
                    "has_text_layer": has_text,
                    "has_tables": has_tables,
                    "has_images": False,
                    "page_count": 1,
                    "text_coverage": 1.0 if has_text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["DOCX 没有可索引文本"],
        )
