"""Agent：单 Agent 主循环的公共入口。

外部只依赖本包导出的 facade/DTO；runtime/ports/adapters 属于内部实现。
``build_default_agent*`` 采用惰性导入，避免仅 ``import agent`` 就实例化全局
Settings 或加载 Provider 依赖。
"""

from .facade import Agent, AgentStream, create_agent
from .public import (
    AgentError,
    AgentErrorCode,
    AgentEvent,
    AgentEventType,
    AgentRequest,
    AgentResult,
    AgentRunView,
    AgentWarning,
    AgentWarningCode,
    CancelReason,
    FinishReason,
    RAGMode,
    RunStatus,
    TurnStatus,
)
from config.agent import AgentConfig

__all__ = [
    # factory / facade
    "create_agent",
    "build_default_agent",
    "build_default_agent_dependencies",
    "Agent",
    "AgentStream",
    # config
    "AgentConfig",
    # public DTO / enums
    "AgentRequest",
    "AgentResult",
    "AgentEvent",
    "AgentRunView",
    "AgentError",
    "AgentWarning",
    "AgentErrorCode",
    "AgentEventType",
    "AgentWarningCode",
    "RunStatus",
    "TurnStatus",
    "CancelReason",
    "FinishReason",
    "RAGMode",
]


def __getattr__(name: str):
    """惰性暴露默认装配入口，保持 ``from agent import ...`` 兼容。"""
    if name == "build_default_agent":
        from .bootstrap import build_default_agent

        return build_default_agent
    if name == "build_default_agent_dependencies":
        from .bootstrap import build_default_agent_dependencies

        return build_default_agent_dependencies
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
