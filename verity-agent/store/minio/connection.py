"""MinIO 客户端生命周期与连接池。

MinIO Python SDK 是同步的；本模块对外暴露 async 方法，内部通过
``asyncio.to_thread`` 调用，避免阻塞事件循环。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from minio import Minio
from minio.error import S3Error

from config import MinioConfig, settings
from exception import (
    ConfigurationError,
    MinioConnectionError,
    MinioError,
    MinioObjectNotFoundError,
    MinioOperationError,
)

logger = logging.getLogger(__name__)

__all__ = ["MinioConnectionManager", "get_minio", "set_minio", "init_minio", "close_minio"]


def _build_http_client(config: MinioConfig):
    """按配置构建 urllib3 连接池；MinIO SDK 默认客户端没有细粒度超时。"""
    import urllib3

    timeout = urllib3.Timeout(
        connect=config.connect_timeout_s,
        read=config.read_timeout_s,
    )
    retries = urllib3.Retry(
        total=config.max_retries,
        backoff_factor=0.2,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD", "PUT", "DELETE"}),
    )
    return urllib3.PoolManager(
        timeout=timeout,
        retries=retries,
        cert_reqs="CERT_REQUIRED" if config.cert_check else "CERT_NONE",
    )


class MinioConnectionManager:
    """MinIO client 封装；构造不发起网络请求。"""

    def __init__(self, config: MinioConfig | None = None) -> None:
        self._config = config or settings.minio
        if not self._config.endpoint.strip():
            raise ConfigurationError("MinIO endpoint 不能为空")
        self._client: Minio | None = None
        self._lock = asyncio.Lock()

    @property
    def config(self) -> MinioConfig:
        return self._config

    @property
    def client(self) -> Minio:
        if self._client is None:
            raise MinioConnectionError("MinIO 客户端尚未初始化，请先调用 open()")
        return self._client

    async def open(self) -> None:
        """创建客户端；``auto_create_buckets`` 时确保两个 bucket 存在。"""
        async with self._lock:
            if self._client is not None:
                return
            if not self._config.enabled:
                logger.info("MinIO 未启用，跳过初始化")
                return
            self._client = Minio(
                self._config.endpoint,
                access_key=self._config.access_key or None,
                secret_key=self._config.secret_key.get_secret_value() or None,
                secure=self._config.secure,
                region=self._config.region,
                http_client=_build_http_client(self._config),
                cert_check=self._config.cert_check,
            )
            if self._config.auto_create_buckets:
                await self.ensure_buckets()

    async def close(self) -> None:
        async with self._lock:
            client, self._client = self._client, None
            if client is None:
                return
            # MinIO SDK 没有 close()；清空底层连接池
            http_client = getattr(client, "_http", None)
            if http_client is not None and hasattr(http_client, "clear"):
                http_client.clear()

    async def ping(self) -> bool:
        """轻量探活：检查知识库 bucket 是否存在/可访问。"""
        try:
            await self.bucket_exists(self._config.buckets.kb)
        except MinioError:
            raise
        except Exception as exc:  # pragma: no cover - SDK 异常兜底
            raise self._translate(exc) from exc
        return True

    async def bucket_exists(self, bucket: str) -> bool:
        try:
            return bool(await asyncio.to_thread(self.client.bucket_exists, bucket))
        except Exception as exc:
            raise self._translate(exc) from exc

    async def ensure_buckets(self) -> None:
        """幂等创建两个 bucket；只应在 dev/test 打开。"""
        for bucket in (self._config.buckets.kb, self._config.buckets.session):
            try:
                exists = await self.bucket_exists(bucket)
                if not exists:
                    await asyncio.to_thread(self.client.make_bucket, bucket)
                    logger.info("已创建 MinIO bucket: %s", bucket)
            except MinioError:
                raise
            except Exception as exc:
                raise self._translate(exc) from exc

    async def health_check(self) -> dict[str, Any]:
        """返回可观测的健康信息，不抛异常。"""
        try:
            ready = await self.ping()
            return {"ready": ready, "endpoint": self._config.endpoint}
        except Exception as exc:
            return {
                "ready": False,
                "endpoint": self._config.endpoint,
                "error": str(exc),
            }

    def translate_error(self, exc: Exception) -> MinioError:
        """供存储门面复用统一的异常翻译。"""
        return self._translate(exc)

    @staticmethod
    def _translate(exc: Exception) -> MinioError:
        if isinstance(exc, MinioError):
            return exc
        if isinstance(exc, S3Error):
            if exc.code in {"NoSuchKey", "NoSuchObject", "NoSuchBucket", "NotFound"}:
                return MinioObjectNotFoundError("对象", exc.object_name or "")
            return MinioOperationError(
                "MinIO S3 操作失败",
                detail={"code": exc.code, "resource": exc.resource},
            )
        import urllib3

        if isinstance(
            exc,
            (
                urllib3.exceptions.HTTPError,
                urllib3.exceptions.MaxRetryError,
                TimeoutError,
                ConnectionError,
                OSError,
            ),
        ):
            return MinioConnectionError("MinIO 连接失败", detail={"cause": str(exc)})
        return MinioOperationError("MinIO 操作失败", detail={"cause": str(exc)})


_default_minio: MinioConnectionManager | None = None


def get_minio() -> MinioConnectionManager:
    global _default_minio
    if _default_minio is None:
        _default_minio = MinioConnectionManager()
    return _default_minio


def set_minio(manager: MinioConnectionManager | None) -> None:
    global _default_minio
    _default_minio = manager


async def init_minio() -> MinioConnectionManager:
    manager = get_minio()
    await manager.open()
    return manager


async def close_minio() -> None:
    global _default_minio
    if _default_minio is not None:
        await _default_minio.close()
        _default_minio = None
