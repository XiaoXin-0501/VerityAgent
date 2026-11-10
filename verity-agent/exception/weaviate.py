"""Weaviate 向量库异常。"""

from __future__ import annotations

from .base import VerityError

__all__ = [
    "WeaviateError",
    "WeaviateConnectionError",
    "WeaviateSchemaError",
    "WeaviateOperationError",
]


class WeaviateError(VerityError):
    """Weaviate 相关异常基类。"""

    code = "WEAVIATE_ERROR"
    http_status = 500
    default_message = "向量库操作失败"


class WeaviateConnectionError(WeaviateError):
    """连接不上 Weaviate / gRPC 不可用 / 超时。"""

    code = "WEAVIATE_CONNECTION_ERROR"
    http_status = 503
    default_message = "向量库连接失败"
    retryable = True


class WeaviateSchemaError(WeaviateError):
    """schema 缺失、字段/分词/维度不一致。"""

    code = "WEAVIATE_SCHEMA_ERROR"
    http_status = 500
    default_message = "向量库 schema 校验失败"
    user_visible = False


class WeaviateOperationError(WeaviateError):
    """查询、写入、删除等操作失败。"""

    code = "WEAVIATE_OPERATION_ERROR"
    http_status = 500
    default_message = "向量库操作失败"
