"""图片 PDF 版式解析的通用小工具。"""

from __future__ import annotations

import re
from typing import Any

from .._common import LIST_ITEM_RE, heading_level_from_text, is_heading_text

__all__ = [
    "as_caption",
    "guess_heading_level",
    "html_to_text",
    "looks_like_list",
    "normalize_for_dedup",
]

_HTML_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def as_caption(value: Any) -> str:
    """把 MinerU 的 caption/footnote 字段统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def looks_like_list(text: str) -> bool:
    """判断文本是否像列表项。"""
    value = text.strip()
    if not value:
        return False
    return bool(LIST_ITEM_RE.match(value))


def html_to_text(html: str) -> str:
    """从 HTML 中提取纯文本；优先 bs4，失败时用正则兜底。"""
    if not html:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]

        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except ImportError:
        text = _HTML_TAG.sub(" ", html)
    return _WHITESPACE.sub(" ", text).strip()


def guess_heading_level(text_level: Any, text: str) -> int | None:
    """结合 MinerU text_level 与编号规则推断标题级别。"""
    level: int | None = None
    if text_level is not None:
        try:
            value = int(text_level)
        except (TypeError, ValueError):
            value = 0
        if value >= 1:
            level = min(6, value)
    if level is not None:
        return level
    if is_heading_text(text):
        return heading_level_from_text(text.strip())
    return None


def normalize_for_dedup(text: str) -> str:
    """用于 tile 重叠区域的文本去重。"""
    return _WHITESPACE.sub("", text or "")
