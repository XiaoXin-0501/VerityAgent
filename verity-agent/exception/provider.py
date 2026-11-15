"""Provider 层异常（统一放在 exception 模块）。

Provider adapter 只负责厂商调用；错误在这里定义，LLM 层再映射为 LLMError。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .base import VerityError

__all__ = ["ProviderErrorCode", "ProviderError", "is_retryable"]


class ProviderErrorCode(StrEnum):
    """Provider 层错误码（《Provider 层数据结构设计》§10.1）。"""

    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    RATE_LIMITED = "rate_limited"
    QUOTA_EXCEEDED = "quota_exceeded"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UPSTREAM = "upstream"
    INVALID_REQUEST = "invalid_request"
    MODEL_NOT_FOUND = "model_not_found"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    RESPONSE_INVALID = "response_invalid"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


_RETRYABLE: frozenset[ProviderErrorCode] = frozenset(
    {
        ProviderErrorCode.RATE_LIMITED,
        ProviderErrorCode.TIMEOUT,
        ProviderErrorCode.CONNECTION,
        ProviderErrorCode.UPSTREAM,
        ProviderErrorCode.UNAVAILABLE,
    }
)


def is_retryable(code: ProviderErrorCode) -> bool:
    """判断该错误码是否允许上层重试。"""
    return code in _RETRYABLE


class ProviderError(VerityError):
    """厂商调用失败的统一异常。"""

    http_status = 502
    default_message = "模型服务调用失败"

    def __init__(
        self,
        code: ProviderErrorCode,
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
        """归一错误码、厂商上下文与可重试标记。"""
        self.provider = provider
        self.model = model
        self.provider_code = provider_code
        self.cause = cause
        self.code = f"PROVIDER_{code.value.upper()}"
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
            retryable=is_retryable(code) if retryable is None else retryable,
            retry_after_s=retry_after_s,
        )

    @property
    def error_code(self) -> ProviderErrorCode:
        """从错误码字符串反解 ProviderErrorCode。"""
        prefix = "PROVIDER_"
        raw = self.code[len(prefix):].lower() if self.code.startswith(prefix) else ""
        try:
            return ProviderErrorCode(raw)
        except ValueError:
            return ProviderErrorCode.INTERNAL

    def __repr__(self) -> str:  # pragma: no cover
        """调试表示；只包含 code/provider/model/message。"""
        return (
            f"ProviderError(code={self.code!r}, provider={self.provider!r}, "
            f"model={self.model!r}, message={self.message!r})"
        )
