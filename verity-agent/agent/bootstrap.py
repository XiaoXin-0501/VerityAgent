"""Agent 默认依赖装配（framework）。

把 Provider chat 模型、PG 消息/租约、工具注册表/流水线、ContextAssembler、
记忆画像/队列、RAG 适配器、事实存储、压缩器与事件存储拼成
``AgentDependencies``；调用方负责基础设施的生命周期与 ``registry.init_all()``。
"""

from __future__ import annotations

from typing import Any

from config import settings
from config.agent import AgentConfig
from prompt import PromptId, get_default_prompt_registry
from utils.time import utcnow

from .adapters.facts import WeaviateFactStore
from .adapters.llm import RoutingChatAdapter
from .adapters.memory import PgMemoryProfileAdapter
from .adapters.pg import PgLeaseAdapter, PgMemoryQueueAdapter, PgMessageAdapter
from .adapters.rag import RetrieverRagAdapter
from .adapters.tools import PipelineToolAdapter
from .context import ContextAssembler
from .facade import Agent, create_agent
from .llm_service import ChatService, ModelRegistry
from .runtime.background import BackgroundTaskRunner
from .runtime.event_store import RedisEventStore
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
    history_limit: int | None = None,
    hooks: Any | None = None,
    memory_queue: Any | None = None,
    background_tasks: Any | None = None,
    memory: Any | None = None,
    context: Any | None = None,
    rag_retriever: Any | None = None,
    compressor: Any | None = None,
    facts: Any | None = None,
    event_store: Any | None = None,
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

    registry = tool_registry or build_tool_registry(
        search_kb=settings.tools.search_kb
    )
    pipeline = tool_pipeline or build_tool_pipeline(registry, settings.tools)
    capabilities = getattr(spec, "capabilities", None)
    max_context_tokens = int(
        getattr(capabilities, "max_context_tokens", 8192) or 8192
    )
    assembler = context or ContextAssembler(
        max_context_tokens=max_context_tokens,
        reserved_output_tokens=min(2048, max(128, max_context_tokens // 4)),
    )
    model_registry = ModelRegistry(
        provider_registry,
        default_model_id=target_model_id,
        model_ids=settings.llm.model_ids,
        fallback_model_ids=settings.llm.fallback_model_ids,
    )
    # 未显式注入时使用内置基础 system prompt；lifespan 会注入带领域信息的版本。
    if system_prompt is None:
        system_prompt = get_default_prompt_registry().render_text(
            PromptId.AGENT_SYSTEM,
            assistant_name=settings.app.name,
            current_date=utcnow().date().isoformat(),
        )
    return AgentDependencies(
        llm=ChatService(
            RoutingChatAdapter(
                model_registry,
                default_model_id=target_model_id,
            ),
            fallback_model_ids=settings.llm.fallback_model_ids,
            enable_fallback=settings.llm.enable_model_fallback,
        ),
        model_id=target_model_id,
        messages=PgMessageAdapter(),
        leases=PgLeaseAdapter(),
        tools=PipelineToolAdapter(registry, pipeline, settings.tools),
        services=services,
        system_prompt=system_prompt,
        history_limit=history_limit or settings.agent.history_limit,
        hooks=hooks,
        memory_queue=memory_queue or PgMemoryQueueAdapter(),
        memory=memory if memory is not None else PgMemoryProfileAdapter(),
        context=assembler,
        rag=(
            RetrieverRagAdapter(rag_retriever)
            if rag_retriever is not None
            else None
        ),
        facts=(
            facts
            if facts is not None
            else (
                WeaviateFactStore()
                if settings.memory.facts_enabled and settings.weaviate.enabled
                else None
            )
        ),
        fact_top_k=settings.memory.fact_top_k,
        fact_min_confidence=settings.memory.fact_min_confidence,
        compressor=compressor,
        event_store=(
            event_store
            if event_store is not None
            else (
                RedisEventStore(
                    ttl_s=settings.agent.run_state_ttl_s,
                    poll_timeout_s=settings.agent.remote_cancel_poll_interval_s,
                )
                if settings.agent.event_store_enabled
                else None
            )
        ),
        background_tasks=background_tasks or BackgroundTaskRunner(),
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
