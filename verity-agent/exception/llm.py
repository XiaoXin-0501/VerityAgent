"""LLM 应用层异常（统一放在 exception 模块）。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .base import VerityError
from .provider import ProviderError, ProviderErrorCode

__all__ = ["LLMErrorCode", "LLMError"]


class LLMErrorCode(StrEnum):
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    CONTENT_FILTER = "content_filter"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UPSTREAM = "upstream"
    UNAVAILABLE = "unavailable"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


_RETRYABLE = frozenset(
    {
        LLMErrorCode.RATE_LIMITED,
        LLMErrorCode.TIMEOUT,
        LLMErrorCode.CONNECTION,
        LLMErrorCode.UPSTREAM,
        LLMErrorCode.UNAVAILABLE,
    }
)

_PROVIDER_MAP = {
    ProviderErrorCode.AUTHENTICATION: LLMErrorCode.AUTHENTICATION,
    ProviderErrorCode.PERMISSION: LLMErrorCode.PERMISSION,
    ProviderErrorCode.RATE_LIMITED: LLMErrorCode.RATE_LIMITED,
    ProviderErrorCode.QUOTA_EXCEEDED: LLMErrorCode.QUOTA_EXCEEDED,
    ProviderErrorCode.TIMEOUT: LLMErrorCode.TIMEOUT,
    ProviderErrorCode.CONNECTION: LLMErrorCode.CONNECTION,
    ProviderErrorCode.UPSTREAM: LLMErrorCode.UPSTREAM,
    ProviderErrorCode.INVALID_REQUEST: LLMErrorCode.INVALID_REQUEST,
    ProviderErrorCode.MODEL_NOT_FOUND: LLMErrorCode.MODEL_NOT_FOUND,
    ProviderErrorCode.UNAVAILABLE: LLMErrorCode.UNAVAILABLE,
    ProviderErrorCode.UNSUPPORTED_CAPABILITY: LLMErrorCode.CAPABILITY_UNSUPPORTED,
    ProviderErrorCode.RESPONSE_INVALID: LLMErrorCode.UPSTREAM,
    ProviderErrorCode.CANCELLED: LLMErrorCode.CANCELLED,
    ProviderErrorCode.INTERNAL: LLMErrorCode.INTERNAL,
}

_HTTP_STATUS = {
    LLMErrorCode.AUTHENTICATION: 503,
    LLMErrorCode.PERMISSION: 503,
    LLMErrorCode.RATE_LIMITED: 429,
    LLMErrorCode.QUOTA_EXCEEDED: 429,
    LLMErrorCode.CONTEXT_LENGTH_EXCEEDED: 422,
    LLMErrorCode.INVALID_REQUEST: 422,
    LLMErrorCode.MODEL_NOT_FOUND: 503,
    LLMErrorCode.CAPABILITY_UNSUPPORTED: 422,
    LLMErrorCode.CONTENT_FILTER: 422,
    LLMErrorCode.TIMEOUT: 504,
    LLMErrorCode.CONNECTION: 503,
    LLMErrorCode.UPSTREAM: 502,
    LLMErrorCode.UNAVAILABLE: 503,
    LLMErrorCode.CANCELLED: 408,
    LLMErrorCode.INTERNAL: 500,
}


class LLMError(VerityError):
    default_message = "模型调用失败"

    def __init__(
        self,
        code: LLMErrorCode,
        message: str | None = None,
        *,
        provider: str | None = None,
        model: str | None = None,
        retry_after_s: float | None = None,
        provider_code: str | None = None,
        details: Mapping[str, Any] | None = None,
        cause: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        """归一错误码、HTTP 状态、可重试标记与附加上下文。"""
        self.error_code = code
        self.provider = provider
        self.model = model
        self.provider_code = provider_code
        self.cause = cause
        self.code = (
            "MODEL_UNAVAILABLE"
            if code is LLMErrorCode.UNAVAILABLE
            else code.value.upper()
        )
        self.http_status = _HTTP_STATUS.get(code, 500)
        merged: dict[str, Any] = {}
        if provider:
            merged["provider"] = provider
        if model:
            merged["model"] = model
        if provider_code:
            merged["provider_code"] = provider_code
        if details:
            merged.update(details)
        super().__init__(
            message or self.default_message,
            details=merged,
            retryable=(code in _RETRYABLE) if retryable is None else retryable,
            retry_after_s=retry_after_s,
        )

    @classmethod
    def from_provider(cls, exc: ProviderError) -> "LLMError":
        """把 ProviderError 映射为 LLMError，保留厂商上下文。"""
        code = _PROVIDER_MAP.get(exc.error_code, LLMErrorCode.INTERNAL)
        return cls(
            code,
            exc.message,
            provider=exc.provider,
            model=exc.model,
            retry_after_s=exc.retry_after_s,
            provider_code=exc.provider_code,
            details=exc.details,
            cause=exc.cause,
            retryable=exc.retryable_value,
        )
