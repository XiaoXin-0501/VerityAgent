"""Weaviate 客户端生命周期。

Weaviate v4 客户端是同步的；本模块提供 async 生命周期方法，内部用
``asyncio.to_thread`` 调用，避免阻塞主事件循环。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import weaviate
from weaviate.auth import AuthApiKey
from weaviate.config import AdditionalConfig, Timeout

from config import WeaviateConfig, settings
from exception import (
    ConfigurationError,
    WeaviateConnectionError,
    WeaviateError,
    WeaviateOperationError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "WeaviateConnectionManager",
    "get_weaviate",
    "set_weaviate",
    "init_weaviate",
    "close_weaviate",
]


class WeaviateConnectionManager:
    """Weaviate client 封装；构造不发起网络请求。"""

    def __init__(self, config: WeaviateConfig | None = None) -> None:
        self._config = config or settings.weaviate
        if not self._config.url.strip():
            raise ConfigurationError("Weaviate url 不能为空")
        self._client: weaviate.WeaviateClient | None = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> WeaviateConfig:
        return self._config

    @property
    def client(self) -> weaviate.WeaviateClient:
        if self._client is None:
            raise WeaviateConnectionError("Weaviate 客户端尚未初始化，请先调用 open()")
        return self._client

    async def open(self) -> None:
        async with self._lock:
            if self._client is not None:
                return
            if not self._config.enabled:
                logger.info("Weaviate 未启用，跳过初始化")
                return
            host, port, secure = self._config.http_parts
            api_key = self._config.api_key.get_secret_value()
            auth = AuthApiKey(api_key) if api_key else None
            additional = AdditionalConfig(
                timeout=Timeout(
                    init=self._config.startup_timeout_s,
                    query=self._config.timeout_s,
                    insert=self._config.timeout_s,
                )
            )
            try:
                self._client = await asyncio.to_thread(
                    weaviate.connect_to_custom,
                    http_host=host,
                    http_port=port,
                    http_secure=secure,
                    grpc_host=self._config.grpc_host or host,
                    grpc_port=self._config.grpc_port,
                    grpc_secure=self._config.grpc_secure,
                    headers=dict(self._config.headers),
                    additional_config=additional,
                    auth_credentials=auth,
                    skip_init_checks=self._config.skip_init_checks,
                )
            except Exception as exc:
                raise WeaviateConnectionError(
                    "无法连接 Weaviate",
                    detail={"url": self._config.url, "cause": str(exc)},
                ) from exc

    async def close(self) -> None:
        async with self._lock:
            client, self._client = self._client, None
            if client is not None:
                try:
                    await asyncio.to_thread(client.close)
                except Exception:  # pragma: no cover - 关闭阶段只记录
                    logger.warning("关闭 Weaviate 客户端失败", exc_info=True)

    async def ping(self) -> bool:
        try:
            return bool(await asyncio.to_thread(self.client.is_ready))
        except WeaviateError:
            raise
        except Exception as exc:
            raise WeaviateConnectionError(
                "Weaviate 探活失败", detail={"cause": str(exc)}
            ) from exc

    async def health_check(self) -> dict[str, Any]:
        try:
            return {"ready": await self.ping(), "url": self._config.url}
        except Exception as exc:
            return {"ready": False, "url": self._config.url, "error": str(exc)}

    async def run_sync(self, func, *args, **kwargs):
        """在事件循环外执行同步 client/collection 操作。"""
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except WeaviateError:
            raise
        except Exception as exc:
            raise WeaviateOperationError(
                "Weaviate 操作失败", detail={"cause": str(exc)}
            ) from exc


_default_weaviate: WeaviateConnectionManager | None = None


def get_weaviate() -> WeaviateConnectionManager:
    global _default_weaviate
    if _default_weaviate is None:
        _default_weaviate = WeaviateConnectionManager()
    return _default_weaviate


def set_weaviate(manager: WeaviateConnectionManager | None) -> None:
    global _default_weaviate
    _default_weaviate = manager


async def init_weaviate() -> WeaviateConnectionManager:
    manager = get_weaviate()
    await manager.open()
    return manager


async def close_weaviate() -> None:
    global _default_weaviate
    if _default_weaviate is not None:
        await _default_weaviate.close()
        _default_weaviate = None
