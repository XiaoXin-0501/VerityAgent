"""工具注册表/流水线 -> Agent ToolPort 适配器。"""

from __future__ import annotations

from agent.tools.enums import ToolKind
from agent.tools.models import (
    ToolContext,
    ToolExposureContext,
    ToolResult,
    ToolSelection,
)
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

    def select(
        self,
        permissions: frozenset[str],
        *,
        exposure: ToolExposureContext | None = None,
    ) -> ToolSelection:
        """按权限与本次请求暴露策略生成工具集合。"""
        return self.registry.for_permissions(
            permissions,
            self.routing,
            exposure,
        )

    def activate_group(
        self,
        selection: ToolSelection,
        router_name: str,
        permissions: frozenset[str],
        *,
        exposure: ToolExposureContext | None = None,
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
            exposure,
        )

    async def execute_batch(
        self,
        calls: list,
        context: ToolContext,
        *,
        max_concurrency: int | None = None,
    ) -> list[ToolResult]:
        """委托 ToolPipeline 并行执行；含 router 调用时按顺序展开工具组。

        router 工具没有业务执行器，只有“激活组”的控制语义；普通流水线会
        拒绝它们。这里把 router 调用交给 ``handle_router``，并用调用方传入
        的 ``context.selection`` 生成下一次 LLM 请求应看到的工具集合。
        """
        if context.selection is None or not any(
            self._is_router(call.name) for call in calls
        ):
            return await self.pipeline.execute_batch(
                calls,
                context,
                max_concurrency=max_concurrency,
            )

        # router 与同批其它调用有顺序依赖：先展开组，再执行组内工具。
        # 路由调用通常一次只有一个，顺序执行不会改变常规批量并发路径。
        results: list[ToolResult] = []
        running_selection = context.selection
        for call in calls:
            tool = self.registry.get(call.name)
            timeout = tool.spec.policy.timeout_s if tool else 30.0
            call_context = context.model_copy(
                update={
                    "call_id": call.call_id,
                    "tool_name": call.name,
                    "timeout_s": timeout,
                }
            )
            if self._is_router(call.name):
                result, running_selection = await self.pipeline.handle_router(
                    running_selection,
                    call,
                    call_context,
                )
            else:
                result = await self.pipeline.execute(call, call_context)
            results.append(result)
        return results

    def _is_router(self, name: str) -> bool:
        """判断注册项是否为 router；未知工具交给流水线返回结构化错误。"""
        registered = self.registry.get(name)
        return (
            registered is not None
            and registered.spec.kind is ToolKind.ROUTER
        )
