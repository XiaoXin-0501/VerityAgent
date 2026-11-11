"""Query 规范化与轻量分析（v1 不做 LLM 改写）。"""

from __future__ import annotations

import re
import unicodedata

from utils.text import normalize_whitespace

__all__ = ["normalize_query", "estimate_language"]

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


def normalize_query(query: str) -> str:
    """全半角归一、空白压缩；保留原始大小写用于实体匹配。"""
    value = unicodedata.normalize("NFKC", query)
    return normalize_whitespace(value, keep_newlines=False)


def estimate_language(text: str) -> str | None:
    """粗粒度语言判断：zh / en；混合或无法判断返回 None。"""
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk and cjk >= latin:
        return "zh"
    if latin and latin > cjk:
        return "en"
    return None
