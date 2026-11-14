"""Redis key 约定：可配置 namespace + 固定 key 族 + 可变后缀。

所有 key 都带应用命名空间（默认 ``verity``，由 ``REDIS__NAMESPACE`` 配置），
避免与其他服务共用 Redis 时冲突。
新增 key 族时在 :class:`RedisKeyPrefix` 里加一个成员，不要在业务代码里手写
字符串前缀。

示例::

    from store.redis import RedisKeyPrefix, make_key

    make_key(RedisKeyPrefix.SESSION, session_id)          # <namespace>:session:<id>
    make_key(RedisKeyPrefix.TOOL_CACHE, tool_name, digest) # <namespace>:tool_cache:<tool>:<digest>
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from exception import RedisKeyError


class RedisKeyPrefix(StrEnum):
    """固定 key 族后缀；``make_key`` 统一加 ``RedisConfig.namespace`` 前缀。"""

    SESSION = "session"
    CONVERSATION = "conversation"
    MESSAGE = "message"
    USER = "user"
    MEMORY = "memory"
    CACHE = "cache"
    TOOL_CACHE = "tool_cache"
    TOOL_CIRCUIT = "tool_circuit"
    IDEMPOTENCY = "idempotency"
    LOCK = "lock"
    RATE_LIMIT = "rate_limit"
    QUEUE = "queue"
    STREAM = "stream"
    # Agent 运行期状态 / SSE 重放缓冲
    RUN = "run"
    # Hooks 用量/成本聚合（Redis 仅作短期缓冲；权威在日志/PG 投影）
    USAGE = "usage"
    # Provider 探活结果（只做观测，不自动摘除；D-66）
    PROVIDER_HEALTH = "provider:health"


def _namespace() -> str:
    """读取部署命名空间；默认 verity，兼容全局 Settings 未初始化场景。"""
    try:
        from config import settings

        value = (settings.redis.namespace or "verity").strip().strip(":")
    except Exception:  # noqa: BLE001 - key 构造不能因配置导入失败而崩溃
        value = "verity"
    return value or "verity"


def make_key(prefix: RedisKeyPrefix, *suffix: Any) -> str:
    """拼出 ``prefix:suffix1:suffix2...``。

    - ``suffix`` 可以是 str / int / UUID / bytes 等，统一转成字符串；
    - ``None`` / 空串 / 纯空白后缀直接拒绝，避免生成 ``<namespace>:session:``
      这类难以排查的 key；
    - 不传后缀时返回前缀本身，适合作为集合/哈希的族名。
    """
    if not isinstance(prefix, RedisKeyPrefix):
        raise RedisKeyError(
            "prefix 必须是 RedisKeyPrefix",
            details={"prefix": str(prefix)},
        )

    parts: list[str] = [f"{_namespace()}:{prefix.value}"]
    for index, part in enumerate(suffix):
        if part is None:
            raise RedisKeyError(
                "Redis key 后缀不能为 None",
                details={"prefix": prefix.value, "index": index},
            )
        if isinstance(part, bytes):
            try:
                text = part.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RedisKeyError(
                    "Redis key 后缀不是合法 UTF-8",
                    details={"prefix": prefix.value, "index": index},
                ) from exc
        else:
            text = str(part)
        text = text.strip()
        if not text:
            raise RedisKeyError(
                "Redis key 后缀不能为空",
                details={"prefix": prefix.value, "index": index},
            )
        parts.append(text)

    return ":".join(parts)


__all__ = ["RedisKeyPrefix", "make_key"]
