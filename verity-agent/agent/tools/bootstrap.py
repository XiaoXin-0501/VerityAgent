"""工具模块装配入口。

启动时由 ``api.lifespan`` 调用；内置工具与 Redis 横切后端在这里接线。
``search_kb`` 只有在 RAG Retriever 就绪时才注册，避免把不可用能力暴露给 LLM。
"""

from __future__ import annotations

from typing import Any

from config.tools import SearchKbToolConfig, ToolsConfig

from .builtin import register_conflict_tool, register_search_kb
from .overflow import MinioOverflowWriter
from .pipeline import ToolPipeline
from .redis_state import build_redis_tool_state
from .registry import ToolRegistry

__all__ = ["build_tool_registry", "build_tool_pipeline"]


def build_tool_registry(
    retriever: Any | None = None,
    *,
    conflict_enabled: bool = False,
    search_kb: SearchKbToolConfig | None = None,
) -> ToolRegistry:
    """创建并注册内置工具；启动后注册表只读。"""
    registry = ToolRegistry()
    _register_builtin_tools(
        registry,
        retriever=retriever,
        conflict_enabled=conflict_enabled,
        search_kb=search_kb,
    )
    return registry


def build_tool_pipeline(
    registry: ToolRegistry,
    config: ToolsConfig,
    *,
    cache: Any | None = None,
    idempotency: Any | None = None,
    circuit_breaker: Any | None = None,
    rate_limiter: Any | None = None,
    overflow_writer: Any | None = None,
) -> ToolPipeline:
    """创建执行流水线，并注入 Redis 缓存/幂等/熔断/限流与 MinIO 溢出。"""
    state = build_redis_tool_state()
    return ToolPipeline(
        registry,
        config,
        cache=cache or state["cache"],
        idempotency=idempotency or state["idempotency"],
        circuit_breaker=circuit_breaker or state["circuit_breaker"],
        rate_limiter=rate_limiter or state["rate_limiter"],
        overflow_writer=overflow_writer or MinioOverflowWriter(),
    )


def _register_builtin_tools(
    registry: ToolRegistry,
    *,
    retriever: Any | None = None,
    conflict_enabled: bool = False,
    search_kb: SearchKbToolConfig | None = None,
) -> None:
    """内置工具注册点；工具依赖由调用方注入，不在此 import 全局单例。"""
    register_search_kb(registry, retriever, config=search_kb)
    if conflict_enabled:
        register_conflict_tool(registry)
