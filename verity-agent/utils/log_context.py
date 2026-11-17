"""请求级日志上下文。

HTTP 中间件在请求入口写入 ``request_id``，日志 Filter 读取后注入格式化字段；
后台任务/worker 没有请求上下文时输出 ``-``，不影响日志写入。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

__all__ = [
    "current_request_id",
    "reset_request_id",
    "set_request_id",
]

_request_id: ContextVar[str | None] = ContextVar(
    "verity_request_id",
    default=None,
)


def set_request_id(value: str) -> Token[str | None]:
    """设置当前异步上下文的请求 id；返回用于恢复的 token。"""
    return _request_id.set(value)


def reset_request_id(token: Token[str | None]) -> None:
    """恢复请求入口前的上下文；必须在请求结束时调用。"""
    _request_id.reset(token)


def current_request_id() -> str:
    """当前请求 id；没有请求上下文时返回 ``-``。"""
    return _request_id.get() or "-"
