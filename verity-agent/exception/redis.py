"""Redis 存储层异常。

连接器会把 ``redis.exceptions`` 翻译成这里的类型，上层不需要 import redis：

- :class:`RedisConnectionError`：连不上 / 连接中断 / 超时（HTTP 503）；
- :class:`RedisCommandError`：命令执行失败（HTTP 500）；
- :class:`RedisKeyError`：key 前缀或后缀不合法（HTTP 400）。
"""

from __future__ import annotations

from .base import VerityError


class RedisError(VerityError):
    """Redis 相关异常基类。"""

    code = "redis_error"
    http_status = 500
    default_message = "Redis 操作失败"


class RedisConnectionError(RedisError):
    """连接不上 Redis、连接中断或命令超时。"""

    code = "redis_connection_error"
    http_status = 503
    default_message = "Redis 连接失败"


class RedisCommandError(RedisError):
    """Redis 命令执行失败（参数、类型、服务端错误等）。"""

    code = "redis_command_error"
    http_status = 500
    default_message = "Redis 命令执行失败"


class RedisKeyError(RedisError):
    """Redis key 前缀或后缀不合法。"""

    code = "redis_key_error"
    http_status = 400
    default_message = "Redis key 不合法"


__all__ = [
    "RedisError",
    "RedisConnectionError",
    "RedisCommandError",
    "RedisKeyError",
]
