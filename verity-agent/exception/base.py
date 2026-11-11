"""项目通用异常基类与统一错误信封。

约定：

- 所有可预期异常继承 :class:`VerityError`；
- ``code`` 是稳定的机器可读错误码，统一 UPPER_SNAKE（与《API 与流式契约》
  §0.2 一致）；业务代码只判断 ``code``，不依赖文案；
- ``http_status`` 决定 HTTP 状态码；
- ``retryable`` / ``user_visible`` 进入错误信封，供前端与网关决策；
- 内部细节放 ``details``，不得包含密钥、堆栈、内部地址。

对外 JSON::

    {"error": {"code": "PERMISSION_DENIED", "message": "...",
               "user_visible": true, "retryable": false,
               "details": {}, "request_id": "req_..."}}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "VerityError",
    "ConfigurationError",
    "ValidationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "InvalidStateTransitionError",
    "RateLimitedError",
    "UnavailableError",
    "InternalError",
]


class VerityError(Exception):
    """项目内所有可预期异常的基类。"""

    code: str = "INTERNAL"
    http_status: int = 500
    default_message: str = "服务内部错误"
    user_visible: bool = True
    retryable: bool = False

    def __init__(
        self,
        message: str | None = None,
        *,
        # 对外错误信封中的 details 字段；也是唯一的内部存储
        details: Mapping[str, Any] | None = None,
        user_visible: bool | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.details: dict[str, Any] = dict(details or {})
        if retry_after_s is not None:
            self.details["retry_after_s"] = retry_after_s
        self.retry_after_s = retry_after_s
        if user_visible is not None:
            self.user_visible_value = user_visible
        else:
            self.user_visible_value = self.user_visible
        if retryable is not None:
            self.retryable_value = retryable
        else:
            self.retryable_value = self.retryable
        super().__init__(self.message)

    def to_dict(self, *, request_id: str | None = None) -> dict[str, Any]:
        """转成《API 与流式契约》§0.2 的错误信封。"""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "user_visible": self.user_visible_value,
            "retryable": self.retryable_value,
            "details": dict(self.details),
        }
        if request_id:
            error["request_id"] = request_id
        return {"error": error}

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r})"
        )


class ConfigurationError(VerityError):
    """配置缺失或不合法；不暴露给普通用户。"""

    code = "CONFIGURATION_ERROR"
    http_status = 500
    default_message = "服务配置错误"
    user_visible = False


class ValidationError(VerityError):
    """入参或业务数据校验失败。"""

    code = "INVALID_REQUEST"
    http_status = 422
    default_message = "参数校验失败"


class AuthenticationError(VerityError):
    """未登录或凭证无效。"""

    code = "UNAUTHENTICATED"
    http_status = 401
    default_message = "认证失败"


class PermissionDeniedError(VerityError):
    """已认证但无权限。"""

    code = "PERMISSION_DENIED"
    http_status = 403
    default_message = "没有访问该资源的权限"


class NotFoundError(VerityError):
    """资源不存在（或对当前调用方不可见）。"""

    code = "NOT_FOUND"
    http_status = 404
    default_message = "资源不存在"

    def __init__(
        self,
        resource: str,
        identifier: object | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        message = (
            f"{resource}不存在"
            if identifier is None
            else f"{resource}不存在: {identifier}"
        )
        merged: dict[str, Any] = {"resource": resource}
        if identifier is not None:
            merged["identifier"] = str(identifier)
        if details:
            merged.update(details)
        super().__init__(message, details=merged)


class ConflictError(VerityError):
    """与当前资源状态冲突（重复、并发修改等）。"""

    code = "CONFLICT"
    http_status = 409
    default_message = "资源状态冲突"


class InvalidStateTransitionError(ConflictError):
    """状态机不允许的迁移，例如 pending 直接跳到 completed。"""

    code = "INVALID_STATE_TRANSITION"
    http_status = 409
    default_message = "非法的状态变更"

    def __init__(
        self,
        resource: str,
        current: object,
        target: object,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "resource": resource,
            "current": str(current),
            "target": str(target),
        }
        if details:
            merged.update(details)
        super().__init__(
            f"{resource}状态不能从 {current} 变更为 {target}",
            details=merged,
        )


class RateLimitedError(VerityError):
    code = "RATE_LIMITED"
    http_status = 429
    default_message = "请求过于频繁，请稍后重试"
    retryable = True


class UnavailableError(VerityError):
    code = "UNAVAILABLE"
    http_status = 503
    default_message = "依赖服务暂不可用"
    retryable = True


class InternalError(VerityError):
    code = "INTERNAL"
    http_status = 500
    default_message = "服务内部错误"
    user_visible = False
