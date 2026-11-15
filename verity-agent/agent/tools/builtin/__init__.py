"""内置工具注册入口。"""

from .conflict import ResolveConflictArgs, register_conflict_tool
from .search_kb import SearchKbArgs, register_search_kb, search_kb_executor

__all__ = [
    "SearchKbArgs",
    "register_search_kb",
    "search_kb_executor",
    "ResolveConflictArgs",
    "register_conflict_tool",
]
