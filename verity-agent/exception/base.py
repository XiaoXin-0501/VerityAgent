"""项目通用异常基类。

约定：
- 所有可预期异常都继承 :class:`VerityError`，便于 FastAPI 层统一捕获并转成
  ``{"code": ..., "message": ..., "detail": ...}`` 的响应；
- 每个异常自带 ``code``（机器可读）与 ``http_status``（HTTP 语义），
  业务代码不要靠异常文案做判断，一律看 ``code``。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class VerityError(Exception):
    """项目内所有可预期异常的基类。"""

    code: str = "internal_error"
    http_status: int = 500
    default_message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        self.message = message or self.default_message
        self.detail: dict[str, Any] = dict(detail or {})
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """转成可直接返回给前端的结构。"""
        data: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            data["detail"] = self.detail
        return data

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"{type(self).__name__}(code={self.code!r}, message={self.message!r})"


class ConfigurationError(VerityError):
    """配置缺失或不合法。"""

    code = "configuration_error"
    http_status = 500
    default_message = "服务配置错误"


class ValidationError(VerityError):
    """入参或业务数据校验失败。"""

    code = "validation_error"
    http_status = 400
    default_message = "参数校验失败"


class AuthenticationError(VerityError):
    """未登录或凭证无效。"""

    code = "authentication_failed"
    http_status = 401
    default_message = "认证失败"


class PermissionDeniedError(VerityError):
    """已认证但无权限。"""

    code = "permission_denied"
    http_status = 403
    default_message = "没有访问该资源的权限"


class NotFoundError(VerityError):
    """资源不存在（或对当前调用方不可见）。"""

    code = "not_found"
    http_status = 404
    default_message = "资源不存在"

    def __init__(
        self,
        resource: str,
        identifier: object | None = None,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        message = (
            f"{resource}不存在"
            if identifier is None
            else f"{resource}不存在: {identifier}"
        )
        merged: dict[str, Any] = {"resource": resource}
        if identifier is not None:
            merged["identifier"] = str(identifier)
        merged.update(detail or {})
        super().__init__(message, detail=merged)


class ConflictError(VerityError):
    """与当前资源状态冲突（重复、并发修改等）。"""

    code = "conflict"
    http_status = 409
    default_message = "资源状态冲突"


class InvalidStateTransitionError(ConflictError):
    """状态机不允许的迁移，例如 pending 直接跳到 completed。"""

    code = "invalid_state_transition"
    http_status = 409
    default_message = "非法的状态变更"

    def __init__(
        self,
        resource: str,
        current: object,
        target: object,
        *,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        merged: dict[str, Any] = {
            "resource": resource,
            "current": str(current),
            "target": str(target),
        }
        merged.update(detail or {})
        super().__init__(
            f"{resource}状态不能从 {current} 变更为 {target}",
            detail=merged,
        )
