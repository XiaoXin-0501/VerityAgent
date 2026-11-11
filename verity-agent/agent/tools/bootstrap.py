"""工具模块装配入口。

启动时由 ``api.lifespan`` 调用；具体工具在 ``register_builtin_tools`` 中注册。
当前 v1 骨架先提供空注册表，后续 ``search_kb`` 等工具在此挂载。
"""

from __future__ import annotations

from config.tools import ToolsConfig

from .pipeline import ToolPipeline
from .registry import ToolRegistry

__all__ = ["build_tool_registry", "build_tool_pipeline"]


def build_tool_registry() -> ToolRegistry:
    """创建并注册内置工具；启动后注册表只读。"""
    registry = ToolRegistry()
    _register_builtin_tools(registry)
    return registry


def build_tool_pipeline(
    registry: ToolRegistry,
    config: ToolsConfig,
) -> ToolPipeline:
    """创建执行流水线；Redis 状态后端在后续阶段注入。"""
    return ToolPipeline(registry, config)


def _register_builtin_tools(registry: ToolRegistry) -> None:
    """内置工具注册点。

    工具文档 §4.9 的 ``search_kb`` 依赖 Retriever 实例，等 RAG 门面接入后
    在这里注册；当前保持空注册表，避免把未完成的能力暴露给 LLM。
    """
    return None
