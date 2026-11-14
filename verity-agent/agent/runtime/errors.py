"""Agent 内部异常与公共错误转换。"""

from __future__ import annotations

from typing import Any

from agent.public import AgentError
from agent.runtime.enums import AgentErrorCode

__all__ = ["AgentRuntimeError", "agent_error"]


class AgentRuntimeError(Exception):
    """内部可预期的中止；携带公共 AgentError，由 facade 转成 AgentResult。"""

    def __init__(self, error: AgentError) -> None:
        self.error = error
        super().__init__(error.message)


def agent_error(
    code: AgentErrorCode,
    message: str,
    *,
    retryable: bool = False,
    user_visible: bool = True,
    details: dict[str, Any] | None = None,
    cause: str | None = None,
    partial_result: bool = False,
) -> AgentError:
    """构造脱敏的公共 AgentError。"""
    return AgentError(
        code=code,
        message=message,
        retryable=retryable,
        user_visible=user_visible,
        details=details or {},
        cause=cause,
        partial_result=partial_result,
    )
