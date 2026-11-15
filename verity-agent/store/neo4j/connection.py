"""Neo4j 驱动生命周期。

Neo4j 驱动本身是异步的；本模块提供连接池、会话、读写事务与探活封装。
未安装 ``neo4j`` 包时模块仍可导入，调用 open() 时才报配置错误。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from config import Neo4jConfig, settings
from exception import (
    ConfigurationError,
    Neo4jConnectionError,
    Neo4jError,
    Neo4jQueryError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Neo4jConnectionManager",
    "get_neo4j",
    "set_neo4j",
    "init_neo4j",
    "close_neo4j",
]


def _driver_exceptions():
    """延迟导入 neo4j 异常，避免未安装依赖时 import 失败。"""
    try:
        from neo4j.exceptions import (  # type: ignore[import-not-found]
            Neo4jError as DriverError,
            ServiceUnavailable,
            SessionExpired,
            TransientError,
        )
    except ImportError as exc:  # pragma: no cover - 依赖缺失
        raise ConfigurationError(
            "未安装 neo4j 驱动，请执行 uv sync 或 pip install neo4j"
        ) from exc
    return DriverError, ServiceUnavailable, SessionExpired, TransientError


class Neo4jConnectionManager:
    """Neo4j AsyncDriver 封装；构造不发起网络请求。"""

    def __init__(self, config: Neo4jConfig | None = None) -> None:
        """保存配置并初始化驱动占位。"""
        self._config = config or settings.neo4j
        self._driver = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> Neo4jConfig:
        """返回 Neo4j 配置。"""
        return self._config

    @property
    def driver(self):
        """返回已打开的驱动；未 open 时抛错。"""
        if self._driver is None:
            raise Neo4jConnectionError("Neo4j 驱动尚未初始化，请先调用 open()")
        return self._driver

    async def open(self) -> None:
        """建立 Neo4j 驱动连接。"""
        async with self._lock:
            if self._driver is not None:
                return
            if not self._config.enabled:
                logger.info("Neo4j 未启用，跳过初始化")
                return
            try:
                from neo4j import AsyncGraphDatabase  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError(
                    "未安装 neo4j 驱动，请执行 uv sync 或 pip install neo4j"
                ) from exc
            driver = None
            try:
                driver = AsyncGraphDatabase.driver(
                    self._config.uri,
                    auth=(
                        self._config.user,
                        self._config.password.get_secret_value(),
                    ),
                    max_connection_pool_size=self._config.max_connection_pool_size,
                    connection_timeout=self._config.connection_timeout_s,
                    max_transaction_retry_time=self._config.max_transaction_retry_time_s,
                    max_connection_lifetime=self._config.max_connection_lifetime_s,
                )
                await driver.verify_connectivity()
            except Exception as exc:
                if driver is not None:
                    try:
                        await driver.close()
                    except Exception:  # pragma: no cover - 关闭失败不掩盖原始异常
                        logger.warning("关闭未就绪的 Neo4j 驱动失败", exc_info=True)
                raise Neo4jConnectionError(
                    "无法连接 Neo4j",
                    details={"uri": self._config.uri, "cause": str(exc)},
                ) from exc
            self._driver = driver

    async def close(self) -> None:
        """关闭驱动并重置状态。"""
        async with self._lock:
            driver, self._driver = self._driver, None
            if driver is not None:
                await driver.close()

    async def ping(self) -> bool:
        """探活：执行轻量查询验证连接。"""
        async with self.session() as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
        return bool(record and record["ok"] == 1)

    async def health_check(self) -> dict[str, Any]:
        """返回包含连接状态的健康信息。"""
        try:
            return {"ready": await self.ping(), "uri": self._config.uri}
        except Exception as exc:
            return {"ready": False, "uri": self._config.uri, "error": str(exc)}

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        """打开一个异步 session 的上下文管理器。"""
        try:
            async with self.driver.session(database=self._config.database) as session:
                yield session
        except Exception as exc:
            raise self._translate(exc) from exc

    async def execute_read(self, work: Callable[..., Any], *args: Any) -> Any:
        """在读写事务中执行回调。"""
        try:
            async with self.session() as session:
                return await session.execute_read(work, *args)
        except Neo4jError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def execute_write(self, work: Callable[..., Any], *args: Any) -> Any:
        """在写事务中执行回调。"""
        try:
            async with self.session() as session:
                return await session.execute_write(work, *args)
        except Neo4jError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    async def run(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """执行一条 Cypher 并消费结果；写操作请优先使用 execute_write。"""
        try:
            async with self.session() as session:
                result = await session.run(query, parameters or {})
                return [record.data() async for record in result]
        except Neo4jError:
            raise
        except Exception as exc:
            raise self._translate(exc) from exc

    @staticmethod
    def _translate(exc: Exception) -> Neo4jError:
        """把 Neo4j 驱动异常翻译为 Neo4jError。"""
        if isinstance(exc, Neo4jError):
            return exc
        try:
            driver_error, unavailable, expired, transient = _driver_exceptions()
        except ConfigurationError:
            return Neo4jQueryError("Neo4j 操作失败", details={"cause": str(exc)})
        if isinstance(exc, (unavailable, expired)):
            return Neo4jConnectionError("Neo4j 连接失败", details={"cause": str(exc)})
        if isinstance(exc, (transient, driver_error)):
            return Neo4jQueryError("Neo4j 查询失败", details={"cause": str(exc)})
        return Neo4jQueryError("Neo4j 操作失败", details={"cause": str(exc)})


_default_neo4j: Neo4jConnectionManager | None = None


def get_neo4j() -> Neo4jConnectionManager:
    """返回全局 Neo4j 管理器，不存在时惰性创建。"""
    global _default_neo4j
    if _default_neo4j is None:
        _default_neo4j = Neo4jConnectionManager()
    return _default_neo4j


def set_neo4j(manager: Neo4jConnectionManager | None) -> None:
    """替换或清空全局 Neo4j 管理器。"""
    global _default_neo4j
    _default_neo4j = manager


async def init_neo4j() -> Neo4jConnectionManager:
    """打开全局 Neo4j 管理器并返回。"""
    manager = get_neo4j()
    await manager.open()
    return manager


async def close_neo4j() -> None:
    """关闭并清空全局 Neo4j 管理器。"""
    global _default_neo4j
    if _default_neo4j is not None:
        await _default_neo4j.close()
        _default_neo4j = None
