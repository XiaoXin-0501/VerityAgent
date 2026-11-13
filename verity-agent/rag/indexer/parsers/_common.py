"""Parser 共用辅助：编码、标题判定、表格构造、Page/ParsedDocument 组装。

这些函数只服务于 parser 层，不放 ``utils``；可跨格式复用的文本工具（编码
探测、Markdown 清理、表格渲染）放在 ``utils.text``。
"""

from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from rag.shared.enums import BlockType
from rag.shared.ids import deterministic_block_id
from rag.shared.models import (
    ContentBlock,
    DocumentProfile,
    DocumentSource,
    Page,
    ParsedDocument,
    TableBlock,
)
from utils.text import decode_text, detect_language, table_to_markdown

__all__ = [
    "LIST_ITEM_RE",
    "HeadingStack",
    "decode_text",
    "heading_level_from_text",
    "is_heading_text",
    "make_block",
    "make_page",
    "make_parsed_document",
    "make_table",
]

# 常见列表项标记：-/*/•、1.、1、、1) 等。parser 只负责识别，不强制改写编号。
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[-*+•·▪◦]|\d{1,3}[.、)）]|[（(]\d{1,3}[)）])\s+"
)

# 标题层级推断：
# - 第一章/第二节 → 1 级
# - 一、/二、     → 2 级
# - 1、/1．       → 3 级
# ASCII 的 "1." / "1)" 更像列表项，不作为标题依据。
_CHAPTER_HEADING_RE = re.compile(
    r"^\s*第[一二三四五六七八九十百\d]+[章节篇部分]\s*"
)
_CHINESE_HEADING_RE = re.compile(r"^\s*[一二三四五六七八九十百]{1,3}[、.．]\s*")
_ARABIC_HEADING_RE = re.compile(r"^\s*\d{1,3}[、．]\s*")


def heading_level_from_text(text: str) -> int | None:
    """根据中文编号习惯推断标题级别；无法判断时返回 ``None``。"""
    value = text.strip()
    if _CHAPTER_HEADING_RE.match(value):
        return 1
    if _CHINESE_HEADING_RE.match(value):
        return 2
    if _ARABIC_HEADING_RE.match(value):
        return 3
    return None


def is_heading_text(text: str, *, max_chars: int = 120) -> bool:
    """按编号模式判断一行是否像标题。

    - 限制长度，避免把长段落误判为标题；
    - 以句号/感叹号结尾的通常不是标题；问号保留（面试题标题常以问号结尾）；
    - 只做保守的编号判断，真正标题仍应由 parser 样式/字号优先决定。
    """
    value = text.strip()
    if not value or len(value) > max_chars:
        return False
    if value.endswith(("。", "！", ".", "!")) and not value.endswith("..."):
        return False
    return heading_level_from_text(value) is not None


class HeadingStack:
    """维护文档当前标题路径的栈。

    parser 每遇到一个标题就调用 ``apply``，后续内容块通过 ``current``
    取得“一级 > 二级 > 三级”的完整路径，而不是只有最近一级标题。
    """

    def __init__(self) -> None:
        """初始化空栈。"""
        self._items: list[tuple[int, str]] = []

    def apply(self, level: int, text: str) -> tuple[str, list[str]]:
        """压入新标题：先弹出同级/更深的旧标题，再返回完整路径。"""
        resolved_level = max(1, min(6, int(level or 1)))
        while self._items and self._items[-1][0] >= resolved_level:
            self._items.pop()
        title = text.strip()
        if title:
            self._items.append((resolved_level, title))
        return self.text() or "", self.path()

    def current(self) -> tuple[str | None, list[str]]:
        """返回当前完整标题文本与路径；没有标题时返回 ``(None, [])``。"""
        return self.text(), self.path()

    def path(self) -> list[str]:
        """返回从一级到当前级的标题文本列表。"""
        return [title for _, title in self._items]

    def text(self) -> str | None:
        """返回用 `` > `` 连接的完整标题文本。"""
        path = self.path()
        return " > ".join(path) if path else None

    @property
    def level(self) -> int | None:
        """返回当前最深标题级别；没有标题时返回 ``None``。"""
        return self._items[-1][0] if self._items else None


def _normalize_cell(value: object) -> str:
    """把任意单元格值转成单行字符串，去首尾空白。"""
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def _column_is_empty(
    index: int,
    headers: list[str],
    rows: list[list[str]],
) -> bool:
    """判断某一列的表头与所有数据单元格是否都为空。"""
    if index < len(headers) and headers[index]:
        return False
    return not any(index < len(row) and row[index] for row in rows)


def _trim_empty_columns(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    """删除所有行都为空的尾部列，避免 Excel 的空白列污染表头。"""
    width = max([len(headers), *(len(row) for row in rows)], default=0)
    while width > 0 and _column_is_empty(width - 1, headers, rows):
        width -= 1
    if width == 0:
        return [], []
    padded_headers = [
        _normalize_cell(headers[index]) if index < len(headers) else ""
        for index in range(width)
    ]
    padded_rows = [
        [
            _normalize_cell(row[index]) if index < len(row) else ""
            for index in range(width)
        ]
        for row in rows
    ]
    return padded_headers, padded_rows


def make_table(
    headers: list[str] | None,
    rows: list[list[str]] | None,
    *,
    caption: str | None = None,
    summary: str | None = None,
    confidence: float | None = None,
) -> TableBlock:
    """构造统一的 ``TableBlock``：补齐列数、清理空列并生成 Markdown。"""
    normalized_headers, normalized_rows = _trim_empty_columns(
        list(headers or []),
        [list(row) for row in (rows or [])],
    )
    return TableBlock(
        caption=caption,
        summary=summary,
        headers=normalized_headers,
        rows=normalized_rows,
        markdown=table_to_markdown(normalized_headers, normalized_rows),
        confidence=confidence,
    )


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
    """按确定性规则构造内容块。

    ``block_id`` 由 ``doc_id + page_no + order + block_type`` 决定，保证同一
    文档重复解析得到同一组 block id。
    """
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


def make_page(
    *,
    page_no: int,
    blocks: list[ContentBlock],
    width: float | None = None,
    height: float | None = None,
    rotation: int | None = None,
    warnings: list[str] | None = None,
) -> Page:
    """根据本页块生成 ``Page``；页面纯文本按阅读顺序拼接。"""
    return Page(
        page_no=page_no,
        width=width,
        height=height,
        rotation=rotation,
        text="\n".join(block.text for block in blocks if block.text),
        blocks=blocks,
        warnings=list(warnings or []),
    )


def _group_blocks_by_page(blocks: list[ContentBlock]) -> list[Page]:
    """把每个块归入其 ``page_no``；保持块在文档中的原始顺序。"""
    grouped: dict[int, list[ContentBlock]] = {}
    for block in blocks:
        grouped.setdefault(block.page_no, []).append(block)
    return [make_page(page_no=page_no, blocks=items) for page_no, items in grouped.items()]


def make_parsed_document(
    *,
    source: DocumentSource,
    doc_id: UUID,
    kb_id: UUID,
    parser_id: str,
    parser_version: str,
    profile: DocumentProfile,
    blocks: list[ContentBlock],
    pages: list[Page] | None = None,
    warnings: list[str] | None = None,
) -> ParsedDocument:
    """组装 ``ParsedDocument``，并补齐画像中的语言信息。

    - ``pages`` 为空时按 ``page_no`` 自动分组，保证每条解析结果都有页结构；
    - 显式声明语言优先；否则从正文粗略检测，写入 ``profile.languages``。
    """
    resolved_pages = pages if pages is not None else _group_blocks_by_page(blocks)
    languages = list(profile.languages)
    if not languages:
        detected = detect_language("\n".join(block.text for block in blocks))
        if detected:
            languages = [detected]
    resolved_profile = profile.model_copy(update={"languages": languages})
    return ParsedDocument(
        doc_id=doc_id,
        kb_id=kb_id,
        parser_id=parser_id,
        parser_version=parser_version,
        profile=resolved_profile,
        pages=resolved_pages,
        blocks=blocks,
        warnings=list(warnings or []),
    )
