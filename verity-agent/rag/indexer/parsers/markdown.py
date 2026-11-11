"""Markdown parser：标题、段落、列表、表格、代码围栏。"""

from __future__ import annotations

import re
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType, SymbolType
from rag.shared.models import (
    CodeBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import decode_text, make_block, make_parsed_document

__all__ = ["MarkdownParser"]

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_FENCE = re.compile(r"^```([\w+-]*)\s*$")
_LIST = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# v1 不处理图片：图片语法从文本中移除，图片-only 行直接跳过。
_IMAGE = re.compile(r"!\[[^\]]*\](?:\([^)]*\)|\[[^\]]*\])")


def _strip_images(text: str) -> str:
    return re.sub(r"\s+", " ", _IMAGE.sub("", text)).strip()


class MarkdownParser:
    parser_id = "markdown"
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
        lines = decode_text(data).replace("\r\n", "\n").replace("\r", "\n").split("\n")
        blocks = []
        order = 0
        index = 0
        heading_text: str | None = None
        has_code = False
        has_tables = False
        has_images = False

        while index < len(lines):
            raw_line = lines[index]
            if not raw_line.strip():
                index += 1
                continue

            fence = _FENCE.match(raw_line)
            if fence:
                line = raw_line
            else:
                if _IMAGE.search(raw_line):
                    has_images = True
                line = _strip_images(raw_line)
                if not line:
                    index += 1
                    continue
            if fence:
                language = fence.group(1).strip() or None
                body: list[str] = []
                index += 1
                while index < len(lines) and not lines[index].startswith("```"):
                    body.append(lines[index])
                    index += 1
                index += 1
                has_code = True
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.CODE,
                        text="\n".join(body),
                        parser_id=self.parser_id,
                        code=CodeBlock(
                            language=language,
                            body="\n".join(body),
                            symbol_type=SymbolType.BLOCK,
                        ),
                        heading_text=heading_text,
                    )
                )
                order += 1
                continue

            heading = _HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                heading_text = title
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.HEADING,
                        text=title,
                        parser_id=self.parser_id,
                        heading_level=level,
                        heading_text=title,
                    )
                )
                order += 1
                index += 1
                continue

            if line.lstrip().startswith("|") and index + 1 < len(lines) and re.match(
                r"^\s*\|?\s*:?-+", lines[index + 1]
            ):
                table_lines = [line]
                index += 1
                while index < len(lines) and lines[index].lstrip().startswith("|"):
                    raw_table_line = lines[index]
                    if _IMAGE.search(raw_table_line):
                        has_images = True
                    table_lines.append(_strip_images(raw_table_line))
                    index += 1
                table = self._table(table_lines)
                has_tables = True
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.TABLE,
                        text="\n".join(table_lines),
                        parser_id=self.parser_id,
                        table=table,
                        heading_text=heading_text,
                    )
                )
                order += 1
                continue

            if _LIST.match(line):
                items = []
                while index < len(lines) and _LIST.match(lines[index]):
                    items.append(_strip_images(_LIST.sub("", lines[index])))
                    index += 1
                text = "\n".join(f"- {item}" for item in items)
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.LIST,
                        text=text,
                        parser_id=self.parser_id,
                        heading_text=heading_text,
                    )
                )
                order += 1
                continue

            paragraph = [line.strip()]
            index += 1
            while index < len(lines):
                raw_candidate = lines[index]
                if (
                    not raw_candidate.strip()
                    or _FENCE.match(raw_candidate)
                    or _HEADING.match(raw_candidate)
                    or _LIST.match(raw_candidate)
                    or raw_candidate.lstrip().startswith("|")
                ):
                    break
                if _IMAGE.search(raw_candidate):
                    has_images = True
                candidate = _strip_images(raw_candidate)
                index += 1
                if candidate:
                    paragraph.append(candidate)
            blocks.append(
                make_block(
                    doc_id=doc_id,
                    page_no=0,
                    order=order,
                    block_type=BlockType.PARAGRAPH,
                    text=" ".join(paragraph),
                    parser_id=self.parser_id,
                    heading_text=heading_text,
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
                    "doc_type": DocumentType.MARKDOWN,
                    "has_text_layer": bool(blocks),
                    "has_code": has_code,
                    "has_tables": has_tables,
                    "has_images": has_images,
                    "page_count": 1,
                    "text_coverage": 1.0 if blocks else 0.0,
                }
            ),
            blocks=blocks,
            warnings=[] if blocks else ["Markdown 文件没有可索引内容"],
        )

    @staticmethod
    def _table(lines: list[str]) -> TableBlock:
        def cells(line: str) -> list[str]:
            return [
                _strip_images(item)
                for item in line.strip().strip("|").split("|")
            ]

        headers = cells(lines[0])
        rows = [cells(item) for item in lines[2:]]
        markdown = "\n".join(lines)
        return TableBlock(headers=headers, rows=rows, markdown=markdown)
