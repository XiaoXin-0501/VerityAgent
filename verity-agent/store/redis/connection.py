"""Redis 异步连接管理（基于 ``redis.asyncio``）。

- 连接池由 ``redis-py`` 管理，默认 ``decode_responses=True``（存 JSON/字符串）；
- 首次 ``open()`` 会 ``PING`` 验证连通性；失败统一抛 ``RedisConnectionError``；
- 提供 ``get/set/delete/expire/ttl/incr/hash/list/set/publish`` 等薄封装；
  ``client`` 属性用于需要高级命令的场景（此时异常不做翻译）；
- 所有命令异常翻译成 ``exception.redis`` 里的类型，上层不需要 import redis。

典型用法::

    from store.redis import init_redis, close_redis, get_redis

    await init_redis()
    redis = get_redis()
    await redis.set(f"{settings.redis.namespace}:cache:foo", "bar", ex=60)
    await close_redis()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from redis.asyncio import Redis
from redis.exceptions import (
    ConnectionError as RedisConnectionException,
    RedisError as RedisException,
    TimeoutError as RedisTimeoutException,
)

from exception import (
    ConfigurationError,
    RedisCommandError,
    RedisConnectionError,
)
from config import RedisConfig, settings

logger = logging.getLogger(__name__)

_CLIENT_NAME = "verity-agent-redis"


class RedisConnectionManager:
    """Redis 客户端封装。

    不传 ``config`` 时使用全局 ``settings.redis``（``REDIS__URL``）。
    客户端在 ``open()`` 时创建并探活，支持显式生命周期管理；业务方法也会
    惰性调用 ``_ensure_open()``，因此在 FastAPI lifespan 里忘记初始化时仍能工作。
    """

    def __init__(
        self,
        config: RedisConfig | None = None,
        *,
        decode_responses: bool | None = None,
        socket_connect_timeout: float | None = None,
        socket_timeout: float | None = None,
        health_check_interval: int | None = None,
        max_connections: int | None = None,
    ) -> None:
        """保存配置与连接参数，构建客户端占位。"""
        self._config = config or settings.redis
        self._decode_responses = (
            self._config.decode_responses
            if decode_responses is None
            else decode_responses
        )
        self._socket_connect_timeout = (
            self._config.socket_connect_timeout
            if socket_connect_timeout is None
            else socket_connect_timeout
        )
        self._socket_timeout = (
            self._config.socket_timeout if socket_timeout is None else socket_timeout
        )
        self._health_check_interval = (
            self._config.health_check_interval
            if health_check_interval is None
            else health_check_interval
        )
        self._max_connections = (
            self._config.max_connections if max_connections is None else max_connections
        )

        if not self._config.url.strip():
            raise ConfigurationError("Redis URL 不能为空")
        if self._socket_connect_timeout <= 0:
            raise ValueError("socket_connect_timeout 必须 > 0")
        if self._socket_timeout <= 0:
            raise ValueError("socket_timeout 必须 > 0")
        if self._health_check_interval < 0:
            raise ValueError("health_check_interval 不能为负数")
        if self._max_connections < 1:
            raise ValueError("max_connections 必须 >= 1")

        self._client: Redis | None = None
        self._lock = asyncio.Lock()

    # ——————————————————— 基本信息 ———————————————————

    @property
    def url(self) -> str:
        """返回 Redis 连接 URL。"""
        return self._config.url

    @property
    def _safe_target(self) -> str:
        """日志/异常里使用的脱敏目标（不包含密码）。"""
        parts = urlsplit(self._config.url)
        host = parts.hostname or "localhost"
        port = parts.port or 6379
        db = parts.path.lstrip("/") or "0"
        return f"{host}:{port}/{db}"

    @property
    def client(self) -> Redis:
        """底层 ``redis.asyncio.Redis``。

        高级命令可直接使用；注意它绕过本类的方法级异常翻译。
        """
        if self._client is None:
            raise RedisConnectionError(
                "Redis 客户端尚未初始化，请先调用 open()/init_redis()",
                details={"target": self._safe_target},
            )
        return self._client

    # ——————————————————— 生命周期 ———————————————————

    async def open(self) -> None:
        """创建客户端并 ``PING`` 探活；重复调用幂等。"""
        async with self._lock:
            if self._client is not None:
                return

            auth_kwargs = {}
            if self._config.username:
                auth_kwargs["username"] = self._config.username
            if self._config.password.get_secret_value():
                auth_kwargs["password"] = self._config.password.get_secret_value()
            try:
                client = Redis.from_url(
                    self._config.url,
                    decode_responses=self._decode_responses,
                    socket_connect_timeout=self._socket_connect_timeout,
                    socket_timeout=self._socket_timeout,
                    health_check_interval=self._health_check_interval,
                    max_connections=self._max_connections,
                    client_name=_CLIENT_NAME,
                    **auth_kwargs,
                )
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    "Redis URL/连接参数不合法",
                    details={"target": self._safe_target, "cause": str(exc)},
                ) from exc

            try:
                await client.ping()
            except (RedisException, OSError) as exc:
                try:
                    await client.aclose()
                except Exception:  # pragma: no cover - 关闭失败不掩盖原始异常
                    logger.warning("关闭未就绪的 Redis 客户端失败", exc_info=True)
                raise RedisConnectionError(
                    f"无法连接 Redis: {exc}",
                    details={"target": self._safe_target, "cause": str(exc)},
                ) from exc

            self._client = client
            logger.info("Redis 已连接: %s", self._safe_target)

    async def close(self) -> None:
        """关闭客户端并释放连接池。"""
        async with self._lock:
            client, self._client = self._client, None
            if client is None:
                return
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - 关闭阶段的异常只记录
                logger.warning("关闭 Redis 客户端失败", exc_info=True)

    async def ping(self) -> bool:
        """探活；失败抛 ``RedisConnectionError``。"""
        await self._call("ping")
        return True

    def get_stats(self) -> dict[str, Any]:
        """同步返回客户端连接池统计（不触发网络 I/O）。"""
        return {
            "target": self._safe_target,
            "decode_responses": self._decode_responses,
            "max_connections": self._max_connections,
        }

    # ——————————————————— 基础命令 ———————————————————

    async def get(self, key: str) -> Any:
        """GET：读取字符串/序列化值。"""
        return await self._call("get", key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        ex: int | None = None,
        px: int | None = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool | None:
        """SET：设置值，支持过期与 NX/XX 条件。"""
        return await self._call("set", key, value, ex=ex, px=px, nx=nx, xx=xx)

    async def delete(self, *keys: str) -> int:
        """DEL：删除一个或多个 key，返回删除数量。"""
        return await self._call("delete", *keys)

    async def exists(self, *keys: str) -> int:
        """EXISTS：返回存在的 key 数量。"""
        return await self._call("exists", *keys)

    async def expire(self, key: str, seconds: int) -> bool:
        """EXPIRE：设置 key 过期秒数。"""
        return await self._call("expire", key, seconds)

    async def ttl(self, key: str) -> int:
        """TTL：返回剩余过期秒数。"""
        return await self._call("ttl", key)

    async def incr(self, key: str, amount: int = 1) -> int:
        """INCRBY：原子递增并返回新值。"""
        return await self._call("incr", key, amount)

    # ——————————————————— Hash ———————————————————

    async def hget(self, name: str, key: str) -> Any:
        """HGET：读取 hash 单个字段。"""
        return await self._call("hget", name, key)

    async def hset(
        self,
        name: str,
        key: str | None = None,
        value: Any | None = None,
        mapping: Mapping[str, Any] | None = None,
    ) -> int:
        """HSET：写入 hash 字段或批量 mapping。"""
        return await self._call(
            "hset",
            name,
            key=key,
            value=value,
            mapping=mapping,
        )

    async def hgetall(self, name: str) -> dict[str, Any]:
        """HGETALL：读取整个 hash。"""
        return await self._call("hgetall", name)

    async def hincrby(self, name: str, key: str, amount: int = 1) -> int:
        """HINCRBY：原子递增 hash 字段。"""
        return await self._call("hincrby", name, key, amount)

    async def hincrbyfloat(self, name: str, key: str, amount: float) -> float:
        """HINCRBYFLOAT：原子递增浮点 hash 字段。"""
        return await self._call("hincrbyfloat", name, key, amount)

    # ——————————————————— List / Set ———————————————————

    async def lpush(self, name: str, *values: Any) -> int:
        """LPUSH：向列表左侧插入元素。"""
        return await self._call("lpush", name, *values)

    async def lrange(self, name: str, start: int = 0, end: int = -1) -> list[Any]:
        """LRANGE：按区间读取列表。"""
        return await self._call("lrange", name, start, end)

    async def sadd(self, name: str, *values: Any) -> int:
        """SADD：向集合添加成员。"""
        return await self._call("sadd", name, *values)

    async def smembers(self, name: str) -> set[Any]:
        """SMEMBERS：读取集合全部成员。"""
        return await self._call("smembers", name)

    # ——————————————————— Pub/Sub 与原始命令 ———————————————————

    async def publish(self, channel: str, message: Any) -> int:
        """PUBLISH：向频道发布消息。"""
        return await self._call("publish", channel, message)

    async def execute_command(self, *args: Any) -> Any:
        """执行原始 Redis 命令；异常仍会被翻译。"""
        return await self._call("execute_command", *args)

    # ——————————————————— 内部实现 ———————————————————

    async def _ensure_open(self) -> None:
        """确保客户端已连接；未初始化时惰性打开。"""
        if self._client is None:
            await self.open()

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """统一调用 Redis 命令并把异常翻译为 RedisError。"""
        await self._ensure_open()
        try:
            return await getattr(self.client, method)(*args, **kwargs)
        except RedisException as exc:
            raise self._translate(exc, method, args[0] if args else None) from exc
        except OSError as exc:
            raise self._translate(exc, method, args[0] if args else None) from exc

    @staticmethod
    def _translate(
        exc: Exception,
        command: str,
        key: object | None = None,
    ) -> RedisConnectionError | RedisCommandError:
        """把 redis-py 异常翻译为连接错误或命令错误。"""
        details: dict[str, Any] = {"command": command}
        if key is not None:
            details["key"] = str(key)
        details["cause"] = str(exc)

        if isinstance(
            exc,
            (RedisConnectionException, RedisTimeoutException, OSError),
        ):
            return RedisConnectionError("Redis 连接失败", details=details)
        return RedisCommandError("Redis 命令执行失败", details=details)


# ——————————————————— 进程内单例 ———————————————————

_default_redis: RedisConnectionManager | None = None


def get_redis() -> RedisConnectionManager:
    """获取进程内默认连接管理器（惰性创建，不立即连接）。"""
    global _default_redis
    if _default_redis is None:
        _default_redis = RedisConnectionManager()
    return _default_redis


def set_redis(manager: RedisConnectionManager | None) -> None:
    """替换默认管理器（测试或自定义连接参数时使用）。"""
    global _default_redis
    _default_redis = manager


async def init_redis() -> RedisConnectionManager:
    """初始化默认客户端，建议放在 FastAPI lifespan 启动阶段。"""
    redis = get_redis()
    await redis.open()
    return redis


async def close_redis() -> None:
    """关闭默认客户端，建议放在 FastAPI lifespan 关闭阶段。"""
    global _default_redis
    if _default_redis is not None:
        await _default_redis.close()
        _default_redis = None


__all__ = [
    "RedisConnectionManager",
    "get_redis",
    "set_redis",
    "init_redis",
    "close_redis",
]
