"""工具模块异常与错误码（统一放在 exception 包）。

两层含义要区分：

- :class:`ToolException` 是**内部异常**，只在服务端流转；
- ``agent.tools.models.ToolError`` 是**值对象**，会进入 ``ToolResult`` 并回喂 LLM。

流水线负责把 ``ToolException`` 统一转成 ``ToolError``，因此异常永远不要穿透到
Agent 主循环。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Mapping

from .base import VerityError

__all__ = [
    "ToolErrorCode",
    "ToolErrorCategory",
    "TOOL_ERROR_CATEGORIES",
    "TOOL_ERROR_RETRYABLE",
    "ToolException",
    "ToolRegistrationError",
    "ToolNotFoundError",
    "ToolPermissionDeniedError",
    "ToolArgumentValidationError",
    "ToolTimeoutError",
    "ToolCircuitOpenError",
    "ToolRateLimitError",
    "ToolIdempotencyConflictError",
    "ToolExecutionError",
    "ToolResultTooLargeError",
    "ToolCancelledError",
    "ToolConfirmationRequiredError",
    "ToolDisabledError",
]


class ToolErrorCode(StrEnum):
    """工具错误码（《工具模块数据结构设计》§7.2.1）。"""

    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    RATE_LIMITED = "RATE_LIMITED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    UPSTREAM_ERROR = "UPSTREAM_ERROR"
    RESULT_TOO_LARGE = "RESULT_TOO_LARGE"
    CANCELLED = "CANCELLED"
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
    TOOL_DISABLED = "TOOL_DISABLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolErrorCategory(StrEnum):
    """错误大类；用于 LLM 提示与告警聚合。"""

    VALIDATION = "VALIDATION"
    PERMISSION = "PERMISSION"
    POLICY = "POLICY"
    EXECUTION = "EXECUTION"
    SYSTEM = "SYSTEM"


TOOL_ERROR_CATEGORIES: dict[ToolErrorCode, ToolErrorCategory] = {
    ToolErrorCode.INVALID_ARGUMENTS: ToolErrorCategory.VALIDATION,
    ToolErrorCode.PERMISSION_DENIED: ToolErrorCategory.PERMISSION,
    ToolErrorCode.TOOL_NOT_FOUND: ToolErrorCategory.VALIDATION,
    ToolErrorCode.TIMEOUT: ToolErrorCategory.EXECUTION,
    ToolErrorCode.CIRCUIT_OPEN: ToolErrorCategory.POLICY,
    ToolErrorCode.RATE_LIMITED: ToolErrorCategory.POLICY,
    ToolErrorCode.IDEMPOTENCY_CONFLICT: ToolErrorCategory.POLICY,
    ToolErrorCode.UPSTREAM_ERROR: ToolErrorCategory.EXECUTION,
    ToolErrorCode.RESULT_TOO_LARGE: ToolErrorCategory.POLICY,
    ToolErrorCode.CANCELLED: ToolErrorCategory.EXECUTION,
    ToolErrorCode.CONFIRMATION_REQUIRED: ToolErrorCategory.POLICY,
    ToolErrorCode.TOOL_DISABLED: ToolErrorCategory.POLICY,
    ToolErrorCode.INTERNAL_ERROR: ToolErrorCategory.SYSTEM,
}

TOOL_ERROR_RETRYABLE: frozenset[ToolErrorCode] = frozenset(
    {
        ToolErrorCode.TIMEOUT,
        ToolErrorCode.CIRCUIT_OPEN,
        ToolErrorCode.RATE_LIMITED,
        ToolErrorCode.UPSTREAM_ERROR,
    }
)

_HTTP_STATUS: dict[ToolErrorCode, int] = {
    ToolErrorCode.INVALID_ARGUMENTS: 422,
    ToolErrorCode.PERMISSION_DENIED: 403,
    ToolErrorCode.TOOL_NOT_FOUND: 404,
    ToolErrorCode.TIMEOUT: 504,
    ToolErrorCode.CIRCUIT_OPEN: 503,
    ToolErrorCode.RATE_LIMITED: 429,
    ToolErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ToolErrorCode.UPSTREAM_ERROR: 502,
    ToolErrorCode.RESULT_TOO_LARGE: 413,
    ToolErrorCode.CANCELLED: 408,
    ToolErrorCode.CONFIRMATION_REQUIRED: 409,
    ToolErrorCode.TOOL_DISABLED: 403,
    ToolErrorCode.INTERNAL_ERROR: 500,
}


class ToolException(VerityError):
    """工具模块内部异常基类。"""

    error_code: ToolErrorCode = ToolErrorCode.INTERNAL_ERROR
    category: ToolErrorCategory = ToolErrorCategory.SYSTEM
    # 注册类异常不在 ToolErrorCode 表里，用固定 code
    code_name: str | None = None
    http_status: int = 500
    default_message: str = "工具执行失败"

    def __init__(
        self,
        message: str | None = None,
        *,
        tool_name: str | None = None,
        code: ToolErrorCode | None = None,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        retry_after_s: float | None = None,
        cause: str | None = None,
    ) -> None:
        """归一工具错误码、类别、HTTP 状态与可重试标记。"""
        if code is not None:
            self.error_code = code
        self.category = TOOL_ERROR_CATEGORIES.get(
            self.error_code, ToolErrorCategory.SYSTEM
        )
        self.tool_name = tool_name
        self.cause = cause
        if self.code_name:
            self.code = self.code_name
        elif self.error_code.value.startswith("TOOL_"):
            self.code = self.error_code.value
        else:
            self.code = f"TOOL_{self.error_code.value}"
        self.http_status = (
            type(self).http_status
            if self.code_name
            else _HTTP_STATUS.get(self.error_code, type(self).http_status)
        )
        merged = dict(details or {})
        if tool_name:
            merged.setdefault("tool_name", tool_name)
        super().__init__(
            message or self.default_message,
            details=merged,
            retryable=(
                self.error_code in TOOL_ERROR_RETRYABLE
                if retryable is None
                else retryable
            ),
            retry_after_s=retry_after_s,
        )


class ToolRegistrationError(ToolException):
    """工具名重复、schema 非法、策略非法或 router_group 不存在。"""

    code_name = "TOOL_REGISTRATION_ERROR"
    http_status = 409
    default_message = "工具注册失败"


class ToolNotFoundError(ToolException):
    """LLM 调用了未注册的工具，或路由组不存在。"""

    error_code = ToolErrorCode.TOOL_NOT_FOUND
    default_message = "工具不存在"

    def __init__(
        self,
        tool_name: str,
        message: str | None = None,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """构造工具不存在错误并带上工具名。"""
        super().__init__(
            message or f"工具不存在: {tool_name}",
            tool_name=tool_name,
            details=details,
        )


class ToolPermissionDeniedError(ToolException):
    """静态能力码或资源级权限校验失败。"""

    error_code = ToolErrorCode.PERMISSION_DENIED
    default_message = "没有调用该工具的权限"


class ToolArgumentValidationError(ToolException):
    """参数 JSON Schema / argument_model 校验失败。"""

    error_code = ToolErrorCode.INVALID_ARGUMENTS
    default_message = "工具参数不合法"


class ToolTimeoutError(ToolException):
    """执行器超时；下游调用应已被取消。"""

    error_code = ToolErrorCode.TIMEOUT
    default_message = "工具执行超时"


class ToolCircuitOpenError(ToolException):
    """工具熔断已打开。"""

    error_code = ToolErrorCode.CIRCUIT_OPEN
    default_message = "工具暂不可用（熔断）"


class ToolRateLimitError(ToolException):
    """工具触发限流。"""

    error_code = ToolErrorCode.RATE_LIMITED
    default_message = "工具调用过于频繁"


class ToolIdempotencyConflictError(ToolException):
    """同一幂等 key 正在执行、状态未知或结果冲突。"""

    error_code = ToolErrorCode.IDEMPOTENCY_CONFLICT
    default_message = "工具幂等冲突"


class ToolExecutionError(ToolException):
    """执行器内部异常或上游依赖失败。

    默认 ``INTERNAL_ERROR``；上游 HTTP/DB 失败时可传
    ``code=ToolErrorCode.UPSTREAM_ERROR``。
    """

    error_code = ToolErrorCode.INTERNAL_ERROR
    default_message = "工具执行失败"


class ToolResultTooLargeError(ToolException):
    """结果超限且无法溢出到 MinIO。"""

    error_code = ToolErrorCode.RESULT_TOO_LARGE
    default_message = "工具结果过大"


class ToolCancelledError(ToolException):
    """调用被取消。"""

    error_code = ToolErrorCode.CANCELLED
    default_message = "工具调用已取消"


class ToolConfirmationRequiredError(ToolException):
    """危险工具需要用户确认；v1 不自动执行。"""

    error_code = ToolErrorCode.CONFIRMATION_REQUIRED
    default_message = "该工具需要用户确认"


class ToolDisabledError(ToolException):
    """工具被禁用或 v1 不允许自动执行写/删除工具。"""

    error_code = ToolErrorCode.TOOL_DISABLED
    default_message = "工具已禁用"
