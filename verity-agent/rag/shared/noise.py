"""实体/关键词噪声判断：版式标签、纯数值、URL/邮箱、长哈希。

图抽取（入库）与图 seed（检索）都需要判断“一个字符串是否值得作为领域实体”；
两处规则必须一致，否则会出现“入库过滤掉、检索又拿来当 seed”的漂移。
"""

from __future__ import annotations

import re

__all__ = [
    "LABEL_RE",
    "NUMERIC_RE",
    "HASH_RE",
    "URL_RE",
    "is_hash_like",
    "is_label_like",
    "is_noise_term",
    "is_numeric_like",
    "is_url_like",
]

# 图/表编号、章节编号、页码等版式标签不是领域概念。
LABEL_RE = re.compile(
    r"^(?:图|表|figure|table|fig\.?|tab\.?)\s*\d+(?:[-.]\d+)*$",
    re.IGNORECASE,
)
# 纯数字/单位/符号串。
NUMERIC_RE = re.compile(r"^[\d\W_]+$", re.UNICODE)
# 8 位以上纯十六进制串（uuid 片段/内容哈希）。
HASH_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)
# URL / 站点地址。
URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def is_label_like(value: str) -> bool:
    """是否图/表/章节编号等版式标签。"""
    return bool(LABEL_RE.match(value or ""))


def is_numeric_like(value: str) -> bool:
    """是否纯数字/单位/符号串。"""
    return bool(NUMERIC_RE.match(value or ""))


def is_hash_like(value: str) -> bool:
    """是否 8 位以上十六进制哈希。"""
    return bool(HASH_RE.match(value or ""))


def is_url_like(value: str) -> bool:
    """是否 URL、邮箱或站点地址。"""
    text = value or ""
    return bool(URL_RE.search(text)) or "@" in text or "://" in text


def is_noise_term(value: str) -> bool:
    """是否为不应进入图/seed 的噪声词。"""
    return (
        is_numeric_like(value)
        or is_label_like(value)
        or is_hash_like(value)
        or is_url_like(value)
    )
