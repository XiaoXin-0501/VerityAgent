"""分页游标工具。

对外 API 使用 cursor 分页（《API 与流式契约》§0.1），存储层仍可用
limit/offset。游标是不透明字符串：base64url(JSON)，客户端不得解析。
"""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from typing import Any

from exception import ValidationError

__all__ = [
    "Page",
    "PageRequest",
    "normalize_limit",
    "encode_cursor",
    "decode_cursor",
]

DEFAULT_LIMIT = 50
MAX_LIMIT = 200


@dataclass(frozen=True, slots=True)
class PageRequest:
    limit: int = DEFAULT_LIMIT
    cursor: str | None = None


@dataclass(frozen=True, slots=True)
class Page[T]:
    items: list[T]
    next_cursor: str | None = None


def normalize_limit(limit: int, *, default: int = DEFAULT_LIMIT, maximum: int = MAX_LIMIT) -> int:
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValidationError("limit 必须是整数")
    if limit <= 0:
        return default
    return min(limit, maximum)


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
        value = json.loads(raw)
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise ValidationError("分页游标不合法") from exc
    if not isinstance(value, dict):
        raise ValidationError("分页游标不合法")
    return value
