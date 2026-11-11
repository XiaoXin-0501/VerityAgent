"""MinIO 对象存储异常。"""

from __future__ import annotations

from .base import NotFoundError, VerityError

__all__ = [
    "MinioError",
    "MinioConnectionError",
    "MinioObjectNotFoundError",
    "MinioOperationError",
]


class MinioError(VerityError):
    """MinIO 相关异常基类。"""

    code = "MINIO_ERROR"
    http_status = 500
    default_message = "对象存储操作失败"


class MinioConnectionError(MinioError):
    """连接不上 MinIO / 连接中断 / 超时。"""

    code = "MINIO_CONNECTION_ERROR"
    http_status = 503
    default_message = "对象存储连接失败"
    retryable = True


class MinioObjectNotFoundError(NotFoundError, MinioError):
    """对象不存在；同时属于 MinIO 错误族，删除场景视为幂等成功。"""

    code = "MINIO_OBJECT_NOT_FOUND"


class MinioOperationError(MinioError):
    """上传、下载、删除、预签名等操作失败。"""

    code = "MINIO_OPERATION_ERROR"
    http_status = 500
    default_message = "对象存储操作失败"
