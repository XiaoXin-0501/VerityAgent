"""哈希与摘要工具。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .serialization import canonical_json

__all__ = [
    "sha256_bytes",
    "sha256_text",
    "sha256_json",
    "sha256_file",
    "stable_digest",
    "short_digest",
]

_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """对规范化 JSON 取 SHA256，用于幂等键/缓存 key。"""
    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*parts: Any) -> str:
    """把多个值按顺序拼成稳定摘要；None 与空串区分保留。"""
    return sha256_json(list(parts))


def short_digest(value: str, length: int = 16) -> str:
    if length < 8 or length > 64:
        raise ValueError("length 必须在 8~64 之间")
    return sha256_text(value)[:length]


def join_digests(values: Iterable[str]) -> str:
    return sha256_text("\x00".join(values))
