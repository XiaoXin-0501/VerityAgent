"""RAG 公共异常（统一放在 exception 模块）。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .base import VerityError

__all__ = ["RagErrorCode", "RagError"]


class RagErrorCode(StrEnum):
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_REQUEST = "INVALID_REQUEST"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL = "INTERNAL"


_HTTP_STATUS = {
    RagErrorCode.INVALID_CONFIG: 500,
    RagErrorCode.INVALID_REQUEST: 422,
    RagErrorCode.PERMISSION_DENIED: 403,
    RagErrorCode.NOT_FOUND: 404,
    RagErrorCode.CONFLICT: 409,
    RagErrorCode.DEPENDENCY_UNAVAILABLE: 503,
    RagErrorCode.INTERNAL: 500,
}

_RETRYABLE = {RagErrorCode.DEPENDENCY_UNAVAILABLE}


class RagError(VerityError):
    """Indexer/Retriever 的公共错误。"""

    default_message = "RAG 操作失败"

    def __init__(
        self,
        code: RagErrorCode,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
        cause: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.error_code = code
        self.cause = cause
        self.code = f"RAG_{code.value}"
        self.http_status = _HTTP_STATUS.get(code, 500)
        merged = dict(details or {})
        super().__init__(
            message or self.default_message,
            details=merged,
            retryable=(code in _RETRYABLE) if retryable is None else retryable,
        )
