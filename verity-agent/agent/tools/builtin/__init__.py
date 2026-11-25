"""内置工具实现入口。

本模块只导出参数模型与执行实现；工具描述、权限、暴露策略统一在
``agent.tools.manifest`` 声明，由 tools bootstrap 装配。
"""

from .conflict import ChunkConflictFactory, ResolveConflictArgs, execute_conflict_tool
from .recall_round import RecallRoundArgs, RecallRoundExecutor, RecallRoundFactory
from .search_kb import (
    SearchKbArgs,
    SearchKbFactory,
    build_search_kb_executor,
    search_kb_executor,
)
from .tool_history import (
    GetToolResultArgs,
    ToolHistoryExecutor,
    ToolHistoryFactory,
)

__all__ = [
    "SearchKbArgs",
    "build_search_kb_executor",
    "search_kb_executor",
    "ResolveConflictArgs",
    "ChunkConflictFactory",
    "execute_conflict_tool",
    "RecallRoundArgs",
    "RecallRoundExecutor",
    "RecallRoundFactory",
    "SearchKbFactory",
    "GetToolResultArgs",
    "ToolHistoryExecutor",
    "ToolHistoryFactory",
]
