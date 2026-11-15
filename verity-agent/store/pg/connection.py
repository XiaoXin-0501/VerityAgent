"""PostgreSQL 异步连接管理（基于官方 ``psycopg_pool.AsyncConnectionPool``）。

- 连接池负责建连、复用、健康检查、坏连接替换、重连与关闭；
- ``transaction()`` / ``connection()`` 使用 psycopg_pool 的标准上下文语义：
  正常退出提交事务，异常退出回滚事务；
- 所有连接统一 ``dict_row``，仓库层拿到的都是列名字典；
- 池在首次取连接时惰性打开，应用启动/关闭阶段可显式调用
  ``init_pg()`` / ``close_pg()``。

典型用法::

    pg = get_pg()
    async with pg.transaction() as conn:
        await conn.execute("UPDATE users SET status = 'active' WHERE id = %s", (uid,))

    # FastAPI lifespan 里
    await init_pg()
    ...
    await close_pg()

.. note::
    Windows 下 psycopg 的异步连接不支持 ``ProactorEventLoop``。uvicorn 在
    ``reload=True`` 或 ``workers > 1`` 时会自动使用 ``SelectorEventLoop``；
    如果单进程无 reload 运行，需要显式设置::

        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolClosed, PoolTimeout

from exception import PGConnectionError
from config import PGConfig, settings

logger = logging.getLogger(__name__)

Row = dict[str, Any]

_POOL_NAME = "verity-agent-pg"


class PGConnectionManager:
    """PostgreSQL 连接池封装。

    所有池参数都来自 :class:`config.PGConfig`（``PG__POOL_*`` /
    ``PG__CONNECT_TIMEOUT``），构造时只需传入 config；不传则使用全局
    ``settings.pg``。对外只暴露 ``open/close/ping/connection/transaction``，
    池的创建推迟到 ``open()``（即在运行中的事件循环里），避免在 import 阶段
    绑定错误的循环。
    """

    def __init__(self, config: PGConfig | None = None) -> None:
        """保存配置并初始化连接池占位。"""
        cfg = config or settings.pg
        self._config = cfg
        self._min_size = cfg.pool_min_size
        self._max_size = cfg.pool_max_size
        self._timeout = cfg.pool_timeout
        self._connect_timeout = cfg.connect_timeout
        self._max_idle = cfg.pool_max_idle
        self._max_lifetime = cfg.pool_max_lifetime

        if self._max_size < 1:
            raise ValueError("pool_max_size 必须 >= 1")
        if not 0 <= self._min_size <= self._max_size:
            raise ValueError("pool_min_size 必须在 0 和 pool_max_size 之间")
        if self._timeout <= 0:
            raise ValueError("pool_timeout 必须 > 0")
        if self._connect_timeout <= 0:
            raise ValueError("connect_timeout 必须 > 0")
        if self._max_idle < 0 or self._max_lifetime < 0:
            raise ValueError("pool_max_idle / pool_max_lifetime 不能为负数")

        self._pool: AsyncConnectionPool[psycopg.AsyncConnection[Row]] | None = None
        self._lock = asyncio.Lock()

    # ——————————————————— 生命周期 ———————————————————

    @property
    def conninfo(self) -> str:
        """返回 psycopg 连接串（由配置字段拼装）。"""
        return make_conninfo(
            host=str(self._config.server),
            port=str(self._config.port),
            user=str(self._config.user),
            password=self._config.password.get_secret_value(),
            dbname=str(self._config.db),
            connect_timeout=str(self._connect_timeout),
        )

    @property
    def pool(self) -> AsyncConnectionPool[psycopg.AsyncConnection[Row]]:
        """底层连接池（必须先 ``open()`` 或用过一次连接）。"""
        if self._pool is None or self._pool.closed:
            raise PGConnectionError("连接池尚未初始化，请先调用 open()/init_pg()")
        return self._pool

    def _build_pool(self) -> AsyncConnectionPool[psycopg.AsyncConnection[Row]]:
        """按配置构建异步连接池。"""
        return AsyncConnectionPool(
            conninfo=self.conninfo,
            min_size=self._min_size,
            max_size=self._max_size,
            timeout=self._timeout,
            max_idle=self._max_idle,
            max_lifetime=self._max_lifetime,
            open=False,
            name=_POOL_NAME,
            kwargs={
                "autocommit": False,
                "row_factory": dict_row,
            },
        )

    async def open(self) -> None:
        """初始化连接池；重复调用幂等。

        ``min_size > 0`` 时会等待首个连接建好（失败抛 ``PGConnectionError``）；
        ``min_size == 0`` 时只启动后台 worker，首次取连接时才真正建连。
        """
        async with self._lock:
            if self._pool is not None and not self._pool.closed:
                return

            pool = self._build_pool()
            try:
                await pool.open(wait=True, timeout=self._connect_timeout)
            except (PoolTimeout, PoolClosed, psycopg.Error) as exc:
                await pool.close()
                raise PGConnectionError(
                    f"无法连接 PostgreSQL {self._config.server}:{self._config.port}"
                    f"/{self._config.db}",
                    details={
                        "host": str(self._config.server),
                        "port": self._config.port,
                        "db": str(self._config.db),
                        "cause": str(exc),
                    },
                ) from exc
            self._pool = pool

    async def close(self) -> None:
        """关闭连接池并等待所有连接释放。"""
        async with self._lock:
            pool, self._pool = self._pool, None
            if pool is not None and not pool.closed:
                await pool.close()

    async def ping(self) -> bool:
        """探活：从池里取一条连接执行 ``SELECT 1``。"""
        try:
            async with self.connection() as conn:
                await conn.execute("SELECT 1")
        except psycopg.Error as exc:
            raise PGConnectionError(
                "PostgreSQL 探活失败",
                details={"host": str(self._config.server), "db": str(self._config.db)},
            ) from exc
        return True

    def get_stats(self) -> dict[str, Any]:
        """连接池运行指标（请求数、连接数、等待数等），用于监控与排障。"""
        return dict(self.pool.get_stats())

    # ——————————————————— 对外上下文 ———————————————————

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection[Row]]:
        """从池里借一条连接。

        遵循 psycopg_pool 的标准语义：正常退出提交、异常退出回滚。
        """
        await self._ensure_open()
        pool = self.pool
        try:
            async with pool.connection() as conn:
                yield conn
        except (PoolTimeout, PoolClosed) as exc:
            raise PGConnectionError(
                "获取 PostgreSQL 连接失败（超时或连接池已关闭）",
                details={
                    "host": str(self._config.server),
                    "db": str(self._config.db),
                    "cause": str(exc),
                },
            ) from exc

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[psycopg.AsyncConnection[Row]]:
        """借一条连接并执行一个事务：成功提交，异常回滚。

        需要跨仓库原子写入时，把该连接作为 ``conn=`` 传给仓库方法即可::

            async with pg.transaction() as conn:
                msg = await msg_repo.append(..., conn=conn)
                await queue_repo.enqueue(..., conn=conn)
        """
        async with self.connection() as conn:
            yield conn

    # ——————————————————— 内部实现 ———————————————————

    async def _ensure_open(self) -> None:
        """确保连接池已打开；未初始化时惰性启动。"""
        if self._pool is None or self._pool.closed:
            await self.open()


# ——————————————————— 进程内单例 ———————————————————

_default_pg: PGConnectionManager | None = None


def get_pg() -> PGConnectionManager:
    """获取进程内默认连接管理器（惰性创建，不立即连接）。"""
    global _default_pg
    if _default_pg is None:
        _default_pg = PGConnectionManager()
    return _default_pg


def set_pg(manager: PGConnectionManager | None) -> None:
    """替换默认管理器（测试或需要自定义池参数时使用）。"""
    global _default_pg
    _default_pg = manager


async def init_pg() -> PGConnectionManager:
    """初始化默认连接池，建议放在 FastAPI lifespan 启动阶段。"""
    pg = get_pg()
    await pg.open()
    return pg


async def close_pg() -> None:
    """关闭默认连接池，建议放在 FastAPI lifespan 关闭阶段。"""
    global _default_pg
    if _default_pg is not None:
        await _default_pg.close()
        _default_pg = None
