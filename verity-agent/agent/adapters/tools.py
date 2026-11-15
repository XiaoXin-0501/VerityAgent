"""工具注册表/流水线 -> Agent ToolPort 适配器。"""

from __future__ import annotations

from agent.tools.enums import ToolKind
from agent.tools.models import ToolContext, ToolResult, ToolSelection
from agent.tools.pipeline import ToolPipeline
from agent.tools.policies import ToolRoutingPolicy
from agent.tools.registry import ToolRegistry
from agent.tools.selection import activate_group
from config.tools import ToolsConfig

__all__ = ["PipelineToolAdapter"]


class PipelineToolAdapter:
    """把现有 ToolRegistry/ToolPipeline 接入 AgentLoop。"""

    def __init__(
        self,
        registry: ToolRegistry,
        pipeline: ToolPipeline,
        config: ToolsConfig,
    ) -> None:
        """保存注册表、流水线与路由配置；不执行任何工具。"""
        self.registry = registry
        self.pipeline = pipeline
        self.routing = ToolRoutingPolicy.from_config(config.routing)

    def select(self, permissions: frozenset[str]) -> ToolSelection:
        """按权限生成一次运行的暴露工具集合。"""
        return self.registry.for_permissions(permissions, self.routing)

    def activate_group(
        self,
        selection: ToolSelection,
        router_name: str,
        permissions: frozenset[str],
    ) -> ToolSelection:
        """router 工具调用后展开组内工具；非 router 返回原 selection。"""
        registered = self.registry.get(router_name)
        if registered is None or registered.spec.kind is not ToolKind.ROUTER:
            return selection
        group_name = registered.spec.router_group
        if not group_name:
            return selection
        return activate_group(
            self.registry.snapshot(),
            selection,
            group_name,
            permissions,
        )

    async def execute_batch(
        self,
        calls: list,
        context: ToolContext,
        *,
        max_concurrency: int | None = None,
    ) -> list[ToolResult]:
        """委托 ToolPipeline 并行执行；流水线负责超时/重试/结果上限。"""
        return await self.pipeline.execute_batch(
            calls,
            context,
            max_concurrency=max_concurrency,
        )
