"""DOCX parser（python-docx，v1 只处理文本与表格，不处理图片）。

改进点：
- 按文档 body 顺序遍历段落和顶层表格，避免“所有段落后再所有表格”的错序；
- 标题同时识别英文 ``Heading N`` 与中文 ``标题 N`` 样式，以及段落大纲级别；
- 列表段落（``List *`` / ``列表*`` 样式或 ``w:numPr``）标记为 LIST；
- 段落文本拼接超链接 run；内嵌图片只计数，不进入 chunk；
- 表格统一走 ``make_table``，生成补全列数的 Markdown 表示。
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterator
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
)

from ._common import HeadingStack, make_block, make_parsed_document, make_table

__all__ = ["DocxParser"]

# 标题样式：Heading 1 / heading 1 / 标题 1。
_HEADING_STYLE = re.compile(r"^(?:heading|标题)\s*(\d+)\s*$", re.IGNORECASE)
# 列表样式：List Bullet / List Number / 列表段落 等。
_LIST_STYLE = re.compile(r"(?:list|列表)", re.IGNORECASE)


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


def _paragraph_text(paragraph) -> str:
    """读取段落文本，包含超链接文本（python-docx 1.2 的 iter_inner_content）。"""
    if hasattr(paragraph, "iter_inner_content"):
        parts = [
            getattr(item, "text", "")
            for item in paragraph.iter_inner_content()
        ]
        return "".join(parts).strip()
    return (paragraph.text or "").strip()


def _heading_level(paragraph) -> int | None:
    """返回标题级别（1~6）；普通段落返回 ``None``。"""
    style = getattr(getattr(paragraph, "style", None), "name", "") or ""
    match = _HEADING_STYLE.match(style.strip())
    if match:
        return max(1, min(6, int(match.group(1))))
    if style.strip().lower() in {"title", "标题"}:
        return 1
    # Word 的大纲级别比样式名更可靠；outlineLvl 从 0 开始。
    p_pr = getattr(paragraph._p, "pPr", None)
    outline = getattr(p_pr, "outlineLvl", None) if p_pr is not None else None
    if outline is not None and outline.val is not None:
        return max(1, min(6, int(outline.val) + 1))
    return None


def _is_list_paragraph(paragraph) -> bool:
    """根据样式或 ``w:numPr`` 判断是否为列表项。"""
    style = getattr(getattr(paragraph, "style", None), "name", "") or ""
    if _LIST_STYLE.search(style):
        return True
    p_pr = getattr(paragraph._p, "pPr", None)
    return p_pr is not None and getattr(p_pr, "numPr", None) is not None


def _table_rows(table) -> list[list[str]]:
    """读取表格为二维字符串；跳过单元格全空的行。"""
    rows = [
        [(cell.text or "").strip() for cell in row.cells]
        for row in table.rows
    ]
    return [row for row in rows if any(cell for cell in row)]


class DocxParser:
    """v1 DOCX 解析器。"""

    parser_id = "docx_python_docx"
    parser_version = "2"

    async def parse(
        self,
        source: DocumentSource,
        profile: DocumentProfile,
        *,
        doc_id: UUID,
        kb_id: UUID,
        data: bytes,
    ) -> ParsedDocument:
        """解析 DOCX 字节为 ``ParsedDocument``。"""
        try:
            from docx import Document
            from docx.table import Table
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
        headings = HeadingStack()
        has_tables = False

        for item in _iter_body_items(document):
            if not isinstance(item, Table):
                # 段落：区分标题 / 列表 / 普通正文。
                text = _paragraph_text(item)
                if not text:
                    continue
                level = _heading_level(item)
                if level is not None:
                    path_text, path = headings.apply(level, text)
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.HEADING,
                            text=text,
                            parser_id=self.parser_id,
                            heading_level=level,
                            heading_text=path_text,
                            heading_path=path,
                        )
                    )
                else:
                    current_heading, current_path = headings.current()
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=(
                                BlockType.LIST
                                if _is_list_paragraph(item)
                                else BlockType.PARAGRAPH
                            ),
                            text=text,
                            parser_id=self.parser_id,
                            heading_text=current_heading,
                            heading_path=current_path,
                        )
                    )
                order += 1
                continue

            rows = _table_rows(item)
            if not rows:
                continue
            has_tables = True
            table = make_table(rows[0], rows[1:])
            current_heading, current_path = headings.current()
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=order,
                    block_type=BlockType.TABLE,
                    text=table.markdown or "",
                    parser_id=self.parser_id,
                    table=table,
                    heading_text=current_heading,
                    heading_path=current_path,
                )
            )
            order += 1

        has_text = bool(blocks)
        # DOCX 无法可靠得到分页信息，统一记为 1 页；图片只记录标志位。
        has_images = bool(getattr(document, "inline_shapes", []))
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
                    "has_images": has_images,
                    "page_count": 1,
                    "text_coverage": 1.0 if has_text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["DOCX 没有可索引文本"],
        )
