"""HTML parser（stdlib）：标题、段落、pre/code、表格。

生产可替换为 trafilatura/readability；本实现保证无额外依赖时也能跑通。
"""

from __future__ import annotations

from html.parser import HTMLParser
from uuid import UUID

from rag.shared.enums import BlockType, DocumentType
from rag.shared.models import (
    CodeBlock,
    DocumentProfile,
    DocumentSource,
    ParsedDocument,
    TableBlock,
)

from ._common import decode_text, make_block, make_parsed_document

__all__ = ["HtmlParser"]

_HEADING_TAGS = {f"h{level}": level for level in range(1, 7)}


class _Collector(HTMLParser):
    """把 HTML 结构流式收集成标题/段落/代码/表格条目。"""

    def __init__(self) -> None:
        """初始化文本缓冲与表格单元格状态。"""
        super().__init__(convert_charrefs=True)
        self.items: list[tuple[str, str, int | None, TableBlock | None]] = []
        self._buffer: list[str] = []
        self._tag: str | None = None
        self._level: int | None = None
        self._table_rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        """遇到结构标签时落盘上一段文本并切换当前上下文。"""
        if tag in _HEADING_TAGS or tag in {"p", "li", "pre", "code"}:
            self._flush()
            self._tag = tag
            self._level = _HEADING_TAGS.get(tag)
        elif tag == "table":
            self._flush()
            self._table_rows = []
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        """闭合标签时完成单元格/行/表格或当前文本条目。"""
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell for cell in self._row):
                self._table_rows.append(self._row)
            self._row = None
        elif tag == "table":
            if self._table_rows:
                table = TableBlock(
                    headers=self._table_rows[0],
                    rows=self._table_rows[1:],
                )
                self.items.append(("table", table.markdown or "", None, table))
            self._table_rows = []
        elif tag == self._tag:
            self._flush()

    def handle_data(self, data: str) -> None:
        """把文本写入当前单元格或当前块缓冲。"""
        if self._cell is not None:
            self._cell.append(data)
        elif self._tag is not None:
            self._buffer.append(data)

    def _flush(self) -> None:
        """把当前缓冲的文本落成一个条目并清空状态。"""
        text = " ".join("".join(self._buffer).split())
        if text and self._tag:
            self.items.append((self._tag, text, self._level, None))
        self._buffer = []
        self._tag = None
        self._level = None


class HtmlParser:
    """HTML 解析器；v1 未注册，保留给后续版本。"""

    parser_id = "html"
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
        """用 stdlib 收集正文/标题/表格，再按阅读顺序构造内容块。"""
        collector = _Collector()
        collector.feed(decode_text(data))
        collector.close()
        blocks = []
        order = 0
        heading_text: str | None = None
        has_code = False
        has_tables = False
        for tag, text, level, table in collector.items:
            if tag == "table" and table is not None:
                has_tables = True
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.TABLE,
                        text=text,
                        parser_id=self.parser_id,
                        table=table,
                        heading_text=heading_text,
                    )
                )
            elif tag in _HEADING_TAGS:
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
            elif tag in {"pre", "code"}:
                has_code = True
                blocks.append(
                    make_block(
                        doc_id=doc_id,
                        page_no=0,
                        order=order,
                        block_type=BlockType.CODE,
                        text=text,
                        parser_id=self.parser_id,
                        code=CodeBlock(body=text),
                        heading_text=heading_text,
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
        return make_parsed_document(
            source=source,
            doc_id=doc_id,
            kb_id=kb_id,
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            profile=profile.model_copy(
                update={
                    "doc_type": DocumentType.HTML,
                    "has_text_layer": True,
                    "has_code": has_code,
                    "has_tables": has_tables,
                    "page_count": 1,
                    "text_coverage": 1.0,
                }
            ),
            blocks=blocks,
        )
