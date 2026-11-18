"""Query 规范化与轻量分析。

意图识别与 query 改写已上移到 Agent 模块（``agent.query_plan``）；RAG 只
保留无模型依赖的规范化、语言判断与 BM25 字段选择。检索请求中的
``rewritten_query`` / ``rag_intent`` 由 Agent 查询规划直接传入。
"""

from __future__ import annotations

import re
import unicodedata

from utils.text import normalize_whitespace

__all__ = [
    "normalize_query",
    "estimate_language",
    "content_field_for_query",
]

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


def content_field_for_query(query: str) -> str:
    """按 query 主语言选择 BM25 字段。

    DocChunk 使用单集合双字段：``content_zh``（gse_ch）与 ``content_en``
    （word）。中文/混合/无法判断默认查中文字段，英文 query 查英文字段。
    """
    return "content_en" if estimate_language(query) == "en" else "content_zh"
