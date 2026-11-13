"""Markdown parser：标题、段落、列表、表格、代码围栏。

v1 约束：
- 标题不单独成 chunk，作为后续块的 ``heading_text`` 上下文；
- 图片语法整体移除（不 OCR、不建 FigureBlock）；
- 行内 Markdown 标记（链接、加粗、行内代码）在正文中清理，避免污染 BM25；
- 代码围栏按纯文本块输出，由 TextChunker 切片（D-76 v1 口径）。
"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, SymbolType
from rag.shared.models import (
    CodeBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
)

from ._common import (
    LIST_ITEM_RE,
    HeadingStack,
    make_block,
    make_parsed_document,
    make_table,
)
from utils.text import (
    decode_text,
    join_wrapped_lines,
    smart_join,
    strip_markdown_inline,
)

__all__ = ["MarkdownParser"]

# ATX 标题：1~6 个 #。
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
# Setext 标题的下划线：=== 或 ---。
_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-+)\s*$")
# 围栏代码块：``` 或 ~~~，允许 3 个以上。
_FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+#.-]*)\s*$")
# 水平分隔线：--- / *** / ___（至少 3 个）。
_HORIZONTAL_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
# 表格分隔行：| --- | :---: | 等。
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)+\|?\s*$")
# YAML front matter 起始行。
_FRONT_MATTER = re.compile(r"^---\s*$")
# 引用块前缀。
_BLOCKQUOTE = re.compile(r"^\s*>\s?")


def _split_table_row(line: str) -> list[str]:
    """按未转义的 ``|`` 拆表格行，并清理行内 Markdown。"""
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith(r"\|"):
        value = value[:-1]
    cells = re.split(r"(?<!\\)\|", value)
    return [
        strip_markdown_inline(cell.replace(r"\|", "|")).strip()
        for cell in cells
    ]


def _is_table_start(lines: list[str], index: int) -> bool:
    """当前行含 ``|`` 且下一行是分隔行时，认为表格开始。"""
    if index + 1 >= len(lines):
        return False
    return "|" in lines[index] and bool(_TABLE_SEPARATOR.match(lines[index + 1]))


def _strip_front_matter(lines: list[str]) -> list[str]:
    """跳过文件头部的 YAML front matter，避免字段被当成正文切片。"""
    if not lines or not _FRONT_MATTER.match(lines[0]):
        return lines
    for index in range(1, min(len(lines), 80)):
        if _FRONT_MATTER.match(lines[index]):
            return lines[index + 1 :]
    return lines


def _collect_paragraph(lines: list[str], index: int) -> tuple[str, int]:
    """从 ``index`` 收集一个段落，直到空行或下一个结构块开始。"""
    collected: list[str] = []
    while index < len(lines):
        raw = lines[index]
        if not raw.strip():
            break
        stripped = raw.strip()
        if (
            _FENCE.match(raw)
            or _ATX_HEADING.match(raw)
            or LIST_ITEM_RE.match(raw)
            or _HORIZONTAL_RULE.match(stripped)
            or _is_table_start(lines, index)
            or (
                index + 1 < len(lines)
                and _SETEXT_UNDERLINE.match(lines[index + 1])
            )
        ):
            break
        collected.append(_BLOCKQUOTE.sub("", raw))
        index += 1
    return join_wrapped_lines([strip_markdown_inline(item) for item in collected]), index


class MarkdownParser:
    """v1 Markdown 解析器（无第三方依赖）。"""

    parser_id = "markdown"
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
        """解析 Markdown 字节为 ``ParsedDocument``。"""
        content = decode_text(data).replace("\r\n", "\n").replace("\r", "\n")
        lines = content.split("\n")
        lines = _strip_front_matter(lines)

        blocks = []
        order = 0
        index = 0
        headings = HeadingStack()
        has_code = False
        has_tables = False
        # 图片语法即使出现在段落/表格/列表续行中也要在画像里体现。
        has_images = "![" in content

        while index < len(lines):
            raw = lines[index]
            if not raw.strip():
                index += 1
                continue
            if "![" in raw:
                has_images = True

            # 1) 围栏代码块：整段作为 CODE 块，语言写入 CodeBlock。
            fence = _FENCE.match(raw)
            if fence:
                marker, language = fence.group(1), fence.group(2) or None
                body: list[str] = []
                index += 1
                while index < len(lines):
                    if lines[index].strip().startswith(marker) and (
                        lines[index].strip()[len(marker):].strip() == ""
                    ):
                        index += 1
                        break
                    body.append(lines[index])
                    index += 1
                has_code = True
                code_text = "\n".join(body).strip("\n")
                current_heading, current_path = headings.current()
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.CODE,
                        text=code_text,
                        parser_id=self.parser_id,
                        code=CodeBlock(
                            language=language,
                            body=code_text,
                            symbol_type=SymbolType.BLOCK,
                        ),
                        heading_text=current_heading,
                        heading_path=current_path,
                    )
                )
                order += 1
                continue

            # 2) Setext 标题：下一行是 === 或 ---。
            if index + 1 < len(lines) and _SETEXT_UNDERLINE.match(lines[index + 1]):
                underline = lines[index + 1].strip()
                level = 1 if underline.startswith("=") else 2
                title = strip_markdown_inline(raw)
                if title:
                    path_text, path = headings.apply(level, title)
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.HEADING,
                            text=title,
                            parser_id=self.parser_id,
                            heading_level=level,
                            heading_text=path_text,
                            heading_path=path,
                        )
                    )
                    order += 1
                index += 2
                continue

            # 3) ATX 标题。
            heading = _ATX_HEADING.match(raw)
            if heading:
                level = len(heading.group(1))
                title = strip_markdown_inline(heading.group(2))
                if title:
                    path_text, path = headings.apply(level, title)
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.HEADING,
                            text=title,
                            parser_id=self.parser_id,
                            heading_level=level,
                            heading_text=path_text,
                            heading_path=path,
                        )
                    )
                    order += 1
                index += 1
                continue

            # 4) 水平分隔线：没有语义，跳过。
            if _HORIZONTAL_RULE.match(raw.strip()):
                index += 1
                continue

            # 5) 表格：当前行 + 分隔行 + 后续 | 行。
            if _is_table_start(lines, index):
                header_cells = _split_table_row(raw)
                index += 2  # 跳过表头与分隔行
                row_cells: list[list[str]] = []
                while index < len(lines) and "|" in lines[index] and lines[index].strip():
                    if "![" in lines[index]:
                        has_images = True
                    row_cells.append(_split_table_row(lines[index]))
                    index += 1
                table = make_table(header_cells, row_cells)
                has_tables = True
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
                continue

            # 6) 列表：保留原标记，缩进续行并入上一项。
            if LIST_ITEM_RE.match(raw):
                items: list[str] = []
                while index < len(lines):
                    current = lines[index]
                    if LIST_ITEM_RE.match(current):
                        if "![" in current:
                            has_images = True
                        items.append(strip_markdown_inline(current.strip()))
                        index += 1
                    elif (
                        current.strip()
                        and current[:1].isspace()
                        and items
                    ):
                        # 缩进续行：并入上一项，避免列表项被拆成独立段落。
                        items[-1] = smart_join(
                            [items[-1], strip_markdown_inline(current.strip())]
                        )
                        index += 1
                    else:
                        break
                if items:
                    current_heading, current_path = headings.current()
                    blocks.append(
                        make_block(
                            doc_id=doc_id,
                            page_no=0,
                            order=order,
                            block_type=BlockType.LIST,
                            text="\n".join(items),
                            parser_id=self.parser_id,
                            heading_text=current_heading,
                            heading_path=current_path,
                        )
                    )
                    order += 1
                continue

            # 7) 普通段落（引用块在这里被去掉 > 前缀）。
            text, index = _collect_paragraph(lines, index)
            if text:
                current_heading, current_path = headings.current()
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.PARAGRAPH,
                        text=text,
                        parser_id=self.parser_id,
                        heading_text=current_heading,
                        heading_path=current_path,
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
                    "doc_type": DocumentType.MARKDOWN,
                    "has_text_layer": has_text,
                    "has_code": has_code,
                    "has_tables": has_tables,
                    "has_images": has_images,
                    "page_count": 1,
                    "text_coverage": 1.0 if has_text else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if has_text else ["Markdown 文件没有可索引内容"],
        )
