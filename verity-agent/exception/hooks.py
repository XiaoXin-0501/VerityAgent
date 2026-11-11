"""Hooks 异常（统一放在 exception 模块）。"""

from __future__ import annotations

from .base import ConflictError, VerityError

__all__ = [
    "HookError",
    "HookRegistrationError",
    "HookTimeoutError",
    "HookExecutionError",
    "HookClosedError",
]


class HookError(VerityError):
    """Hooks 相关异常基类。"""

    code = "HOOK_ERROR"
    http_status = 500
    default_message = "Hook 执行失败"


class HookRegistrationError(ConflictError):
    """Hook 名称重复或注册声明非法。"""

    code = "HOOK_REGISTRATION_ERROR"
    http_status = 409
    default_message = "Hook 注册失败"


class HookTimeoutError(HookError):
    """单个 Hook 执行超时。"""

    code = "HOOK_TIMEOUT"
    http_status = 504
    default_message = "Hook 执行超时"
    retryable = False


class HookExecutionError(HookError):
    """Hook 执行异常；FAIL_CLOSED 时向上抛出。"""

    code = "HOOK_EXECUTION_ERROR"
    http_status = 500
    default_message = "Hook 执行异常"


class HookClosedError(HookError):
    """dispatcher 已关闭，拒绝新事件。"""

    code = "HOOK_CLOSED"
    http_status = 503
    default_message = "Hooks 已关闭"
