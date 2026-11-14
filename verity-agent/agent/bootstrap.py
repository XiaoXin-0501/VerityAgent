"""Agent 默认依赖装配（framework）。

当前只接入已有基础设施：Provider chat 模型、PG 消息/租约、工具注册表。
Memory/Context/压缩/抽取尚未实现，因此 ``system_prompt`` 为空、消息窗口
直接取最近 N 条；后续接入 ContextAssembler 时替换依赖即可。
"""

from __future__ import annotations

from typing import Any

from config import settings
from config.agent import AgentConfig

from .adapters.llm import RoutingChatAdapter
from .adapters.pg import PgLeaseAdapter, PgMessageAdapter
from .adapters.tools import PipelineToolAdapter
from .facade import Agent, create_agent
from .runtime.ports import AgentDependencies
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry

__all__ = ["build_default_agent_dependencies", "build_default_agent"]


def build_default_agent_dependencies(
    *,
    provider_registry: Any,
    model_id: str | None = None,
    tool_registry: Any | None = None,
    tool_pipeline: Any | None = None,
    services: Any | None = None,
    system_prompt: str | None = None,
    history_limit: int = 40,
    hooks: Any | None = None,
) -> AgentDependencies:
    """用当前 Settings/ProviderRegistry 构造 Agent 依赖。

    调用方负责 ``registry.init_all()`` 与 PG 生命周期；这里只拼对象，
    不打开连接。工具注册表为空时 LLM 不会看到工具，但普通对话仍可运行。
    """
    target_model_id = model_id or settings.llm.default_model_id
    if target_model_id is None:
        target_model_id = next(
            (
                item.id
                for item in settings.models
                if item.enabled and item.kind.value == "chat"
            ),
            None,
        )
    if not target_model_id:
        raise ValueError("没有可用的 chat 模型；请配置 MODELS/LLM__DEFAULT_MODEL_ID")
    chat_model = provider_registry.get_model_by_id(target_model_id)
    spec = getattr(chat_model, "spec", None)
    if spec is not None and getattr(spec, "kind", None) is not None:
        if getattr(spec.kind, "value", spec.kind) != "chat":
            raise ValueError(
                f"Agent 主模型必须是 chat 模型: {target_model_id} "
                f"(kind={getattr(spec.kind, 'value', spec.kind)})"
            )

    registry = tool_registry or build_tool_registry()
    pipeline = tool_pipeline or build_tool_pipeline(registry, settings.tools)
    return AgentDependencies(
        llm=RoutingChatAdapter(
            provider_registry,
            default_model_id=target_model_id,
        ),
        model_id=target_model_id,
        messages=PgMessageAdapter(),
        leases=PgLeaseAdapter(),
        tools=PipelineToolAdapter(registry, pipeline, settings.tools),
        services=services,
        system_prompt=system_prompt,
        history_limit=history_limit,
        hooks=hooks,
    )


async def build_default_agent(
    config: AgentConfig | None = None,
    *,
    provider_registry: Any,
    **kwargs: Any,
) -> Agent:
    """构造默认 Agent；依赖由 ``build_default_agent_dependencies`` 提供。"""
    dependencies = build_default_agent_dependencies(
        provider_registry=provider_registry,
        **kwargs,
    )
    return await create_agent(config or AgentConfig(), dependencies=dependencies)
