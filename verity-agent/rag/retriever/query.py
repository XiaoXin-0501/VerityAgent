"""Query 规范化与轻量分析。

规则层只做低成本、可解释的判断：规范化、语言、意图。规则无法判断时，
由 ``rag.retriever.query_llm.ChatIntentClassifier`` 做 LLM 兜底；LLM 失败
不影响检索，保留规则结果/空意图。
"""

from __future__ import annotations

import re
import unicodedata

from rag.shared.enums import QueryIntent
from utils.text import normalize_whitespace

__all__ = [
    "normalize_query",
    "estimate_language",
    "content_field_for_query",
    "classify_intent_rule",
]

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")

# 规则只覆盖高置信表达；模糊 query 交给 LLM 兜底，避免误分类。
_INTENT_RULES: tuple[tuple[QueryIntent, tuple[str, ...]], ...] = (
    (
        QueryIntent.COMPARE,
        (
            "对比", "比较", "区别", "差异", "哪个好", "哪个更",
            "difference", "compare", "versus", " vs ", "vs.", "better",
        ),
    ),
    (
        QueryIntent.SUMMARIZE,
        (
            "总结", "概括", "摘要", "综述", "主要讲", "讲了什么",
            "summarize", "summary", "overview", "key points",
        ),
    ),
    (
        QueryIntent.HOWTO,
        (
            "如何", "怎么", "怎样", "步骤", "流程", "治疗方法", "预防",
            "how to", "how do", "steps", "procedure", "treatment",
        ),
    ),
    (
        QueryIntent.LOOKUP,
        (
            "是什么", "定义", "含义", "what is", "definition", "meaning",
        ),
    ),
)


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


def classify_intent_rule(query: str) -> QueryIntent | None:
    """规则识别高置信 query 意图；无法判断返回 None，交给 LLM 兜底。"""
    value = f" {query.lower()} "
    for intent, needles in _INTENT_RULES:
        if any(needle in value for needle in needles):
            return intent
    return None
