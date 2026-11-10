"""文本清洗与展示工具。"""

from __future__ import annotations

import re

__all__ = [
    "normalize_whitespace",
    "strip_control_chars",
    "truncate",
    "ensure_prefix",
    "mask_secret",
]

_WHITESPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_whitespace(text: str, *, keep_newlines: bool = True) -> str:
    """压缩水平空白；``keep_newlines`` 时保留段落，否则压成单行。"""
    value = _CONTROL.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))
    value = _WHITESPACE.sub(" ", value)
    if keep_newlines:
        value = _MULTI_NEWLINE.sub("\n\n", value)
    else:
        value = value.replace("\n", " ")
    return value.strip()


def strip_control_chars(text: str) -> str:
    return _CONTROL.sub("", text)


def truncate(text: str, max_chars: int, *, suffix: str = "…") -> str:
    if max_chars < 0:
        raise ValueError("max_chars 不能为负数")
    if len(text) <= max_chars:
        return text
    if max_chars <= len(suffix):
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)].rstrip() + suffix


def ensure_prefix(value: str, prefix: str) -> str:
    return value if value.startswith(prefix) else f"{prefix}{value}"


def mask_secret(value: str | None, *, visible: int = 4) -> str:
    """日志/错误里脱敏密钥；短值整体打码。"""
    if not value:
        return ""
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * (len(value) - visible * 2)}{value[-visible:]}"
