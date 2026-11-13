"""纯文本 parser：按空行/换行组织段落，并识别列表与编号标题。

TXT 没有样式信息，这里用保守规则：
- 空行分段；没有空行时按单行处理，但连续正文行会合并成一个段落；
- ``- ``/``1.`` 等列表标记聚合成 LIST 块；
- ``1、``/``第一章`` 等编号短行识别为 HEADING，供后续块的 heading 上下文。

该 parser 同时作为未知格式的 fallback（由 ParserRegistry 决定）。
"""

from __future__ import annotations

from uuid import UUID

from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import DocumentProfile, DocumentSource, ParsedDocument

from ._common import (
    LIST_ITEM_RE,
    HeadingStack,
    decode_text,
    heading_level_from_text,
    is_heading_text,
    make_block,
    make_parsed_document,
)
from utils.text import join_wrapped_lines

__all__ = ["TextParser"]


class TextParser:
    """v1 纯文本解析器。"""

    parser_id = "text"
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
        """解析 TXT 字节为 ``ParsedDocument``。"""
        text = decode_text(data).replace("\r\n", "\n").replace("\r", "\n").strip()
        lines = text.split("\n")

        blocks = []
        order = 0
        headings = HeadingStack()
        paragraph: list[str] = []
        list_items: list[str] = []

        def flush_paragraph() -> None:
            """把缓存的正文行合并成一个段落块。"""
            nonlocal order
            value = join_wrapped_lines(paragraph)
            paragraph.clear()
            if not value:
                return
            current_heading, current_path = headings.current()
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=order,
                    block_type=BlockType.PARAGRAPH,
                    text=value,
                    parser_id=self.parser_id,
                    heading_text=current_heading,
                    heading_path=current_path,
                )
            )
            order += 1

        def flush_list() -> None:
            """把缓存的列表项合并成一个 LIST 块。"""
            nonlocal order
            if not list_items:
                return
            value = "\n".join(list_items)
            list_items.clear()
            current_heading, current_path = headings.current()
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=order,
                    block_type=BlockType.LIST,
                    text=value,
                    parser_id=self.parser_id,
                    heading_text=current_heading,
                    heading_path=current_path,
                )
            )
            order += 1

        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                # 空行是段落边界，也是列表边界。
                flush_paragraph()
                flush_list()
                continue

            if is_heading_text(stripped):
                flush_paragraph()
                flush_list()
                level = heading_level_from_text(stripped) or 1
                path_text, path = headings.apply(level, stripped)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=stripped,
                        parser_id=self.parser_id,
                        heading_level=level,
                        heading_text=path_text,
                        heading_path=path,
                    )
                )
                order += 1
                continue

            if LIST_ITEM_RE.match(raw):
                flush_paragraph()
                list_items.append(stripped)
                continue

            # 正文行：列表结束后先落盘列表，再继续累积段落。
            flush_list()
            paragraph.append(stripped)

        flush_paragraph()
        flush_list()

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
