"""工具服务依赖 key 与通用读取。

工厂和工具执行器统一使用这些 key 从 ``app.state`` / Mapping 读取服务，
避免装配期与执行期各自写字符串导致 key 不一致。
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

__all__ = ["ToolServiceKey", "get_tool_service"]


class ToolServiceKey(StrEnum):
    """工具依赖服务名；与 app.state 属性保持一致。"""

    RAG_RETRIEVER = "rag_retriever"
    CHUNK_CONFLICT_SERVICE = "chunk_conflict_service"
    CONVERSATION_MESSAGES = "conversation_messages"


def get_tool_service(
    services: Any,
    key: ToolServiceKey | str,
    default: Any = None,
) -> Any:
    """从 Mapping 或 app.state 风格对象读取服务；缺失返回默认值。"""
    if services is None:
        return default
    name = key.value if isinstance(key, ToolServiceKey) else str(key)
    if isinstance(services, Mapping):
        return services.get(name, default)
    return getattr(services, name, default)
