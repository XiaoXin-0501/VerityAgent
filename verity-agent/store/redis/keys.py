"""Redis key 约定：固定前缀（enum）+ 可变后缀。

所有 key 都带应用命名空间 ``verity:``，避免与其他服务共用 Redis 时冲突。
新增 key 族时在 :class:`RedisKeyPrefix` 里加一个成员，不要在业务代码里手写
字符串前缀。

示例::

    from store.redis import RedisKeyPrefix, make_key

    make_key(RedisKeyPrefix.SESSION, session_id)          # verity:session:<id>
    make_key(RedisKeyPrefix.TOOL_CACHE, tool_name, digest) # verity:tool_cache:<tool>:<digest>
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from exception import RedisKeyError


class RedisKeyPrefix(StrEnum):
    """固定 key 前缀。值包含 ``verity:`` 命名空间。"""

    SESSION = "verity:session"
    CONVERSATION = "verity:conversation"
    MESSAGE = "verity:message"
    USER = "verity:user"
    MEMORY = "verity:memory"
    CACHE = "verity:cache"
    TOOL_CACHE = "verity:tool_cache"
    IDEMPOTENCY = "verity:idempotency"
    LOCK = "verity:lock"
    RATE_LIMIT = "verity:rate_limit"
    QUEUE = "verity:queue"
    STREAM = "verity:stream"


def make_key(prefix: RedisKeyPrefix, *suffix: Any) -> str:
    """拼出 ``prefix:suffix1:suffix2...``。

    - ``suffix`` 可以是 str / int / UUID / bytes 等，统一转成字符串；
    - ``None`` / 空串 / 纯空白后缀直接拒绝，避免生成 ``verity:session:``
      这类难以排查的 key；
    - 不传后缀时返回前缀本身，适合作为集合/哈希的族名。
    """
    if not isinstance(prefix, RedisKeyPrefix):
        raise RedisKeyError(
            "prefix 必须是 RedisKeyPrefix",
            detail={"prefix": str(prefix)},
        )

    parts: list[str] = [prefix.value]
    for index, part in enumerate(suffix):
        if part is None:
            raise RedisKeyError(
                "Redis key 后缀不能为 None",
                detail={"prefix": prefix.value, "index": index},
            )
        if isinstance(part, bytes):
            try:
                text = part.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RedisKeyError(
                    "Redis key 后缀不是合法 UTF-8",
                    detail={"prefix": prefix.value, "index": index},
                ) from exc
        else:
            text = str(part)
        text = text.strip()
        if not text:
            raise RedisKeyError(
                "Redis key 后缀不能为空",
                detail={"prefix": prefix.value, "index": index},
            )
        parts.append(text)

    return ":".join(parts)


__all__ = ["RedisKeyPrefix", "make_key"]
