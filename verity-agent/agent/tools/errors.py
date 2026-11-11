"""内部异常 -> ToolError 的统一映射。

LLM 只应看到结构化的 ToolError；Python 异常/堆栈最多进入日志的 ``cause``。
"""

from __future__ import annotations

import asyncio

from exception.tool import ToolErrorCode, ToolException, ToolTimeoutError

from .models import ToolError

__all__ = ["tool_error_from_exception"]


def tool_error_from_exception(
    exc: BaseException,
    *,
    tool_name: str | None = None,
) -> ToolError:
    if isinstance(exc, ToolException):
        error = ToolError.from_exception(exc)
        if tool_name and "tool_name" not in error.details:
            return error.model_copy(
                update={"details": {**error.details, "tool_name": tool_name}}
            )
        return error
    if isinstance(exc, asyncio.TimeoutError):
        error = ToolError.from_exception(ToolTimeoutError(tool_name=tool_name))
        return error
    return ToolError(
        code=ToolErrorCode.INTERNAL_ERROR,
        message="工具执行失败",
        details={"tool_name": tool_name} if tool_name else {},
        cause=str(exc),
    )
