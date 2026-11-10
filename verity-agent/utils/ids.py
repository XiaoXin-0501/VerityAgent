"""标识符生成工具。

项目约定（《README》§4）：

- 持久化主键优先 UUIDv7（时间有序，索引友好）；
- 运行期 id（``run_id`` / ``turn_id`` / ``request_id``）是字符串，但同样
  使用 UUIDv7 的十六进制形式，避免另外维护一套 id 规则；
- 对外展示可带类型前缀（``run_`` / ``req_``），解析时只取最后一段。

Python 3.12 没有内置 UUIDv7，这里按 RFC 9562 布局手工生成；同一毫秒内的
随机位足够大，不需要额外的进程内序列号。
"""

from __future__ import annotations

import os
import time
from uuid import UUID

__all__ = [
    "uuid7",
    "new_uuid",
    "new_token",
    "new_request_id",
    "new_run_id",
    "new_turn_id",
    "new_idempotency_key",
]

_UUID7_VERSION = 0x7
_UUID7_VARIANT = 0b10
_UINT48_MASK = (1 << 48) - 1
_UINT12_MASK = (1 << 12) - 1
_UINT62_MASK = (1 << 62) - 1


def uuid7() -> UUID:
    """生成 RFC 9562 UUIDv7。"""
    timestamp_ms = int(time.time() * 1000) & _UINT48_MASK
    rand_a = int.from_bytes(os.urandom(2), "big") & _UINT12_MASK
    rand_b = int.from_bytes(os.urandom(8), "big") & _UINT62_MASK

    value = timestamp_ms << 80
    value |= _UUID7_VERSION << 76
    value |= rand_a << 64
    value |= _UUID7_VARIANT << 62
    value |= rand_b
    return UUID(int=value)


def new_uuid() -> str:
    """UUIDv7 的字符串形式（无前缀）。"""
    return str(uuid7())


def new_token(nbytes: int = 16) -> str:
    """密码学随机的十六进制 token；用于幂等键、一次性口令等。"""
    if nbytes < 8:
        raise ValueError("nbytes 必须 >= 8")
    return os.urandom(nbytes).hex()


def new_request_id() -> str:
    """HTTP 请求 / 运行链路的请求 id，形如 ``req_<uuid7hex>``。"""
    return f"req_{uuid7().hex}"


def new_run_id() -> str:
    """运行 id；PG ``conversations.current_run_id`` 是 UUID，所以不加前缀。"""
    return str(uuid7())


def new_turn_id() -> str:
    """回合 id；运行期字符串，不落库。"""
    return str(uuid7())


def new_idempotency_key() -> str:
    return f"idem_{new_token(16)}"
