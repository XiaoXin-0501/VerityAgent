"""Agent 默认依赖装配（framework）。

把 Provider chat 模型、PG 消息/租约、工具注册表/流水线、ContextAssembler、
记忆画像/队列、RAG 适配器、事实存储、压缩器与事件存储拼成
``AgentDependencies``；调用方负责基础设施的生命周期与 ``registry.init_all()``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import settings
from config.agent import AgentConfig
from config.domain import (
    domain_catalog_text,
    list_domain_profile_names,
    load_domain_profile,
)
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
from .query_plan import QueryPlanner
from .runtime.event_store import RedisEventStore
from .runtime.flow import build_flow_log
from .runtime.ports import AgentDependencies
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry

__all__ = ["build_default_agent_dependencies", "build_default_agent"]

logger = logging.getLogger(__name__)


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
    query_planner: Any | None = None,
    query_plan_config: Any | None = None,
    reflection_model_id: str | None = None,
    flow_log: Any | None = None,
    domain_catalog: str | None = None,
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
        search_kb=settings.tools.search_kb,
        recall_round=settings.tools.recall_round,
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
    # 查询规划/反思可以配置独立小模型；它们同样必须能通过 ModelRegistry
    # 的白名单，否则请求会在路由层被拒绝并静默退化。这里把显式配置的
    # agent 辅助模型并入白名单（LLM model_ids 为空时仍不做限制）。
    agent_model_ids = list(settings.llm.model_ids)
    effective_plan_config = query_plan_config or settings.agent.query_plan
    for model_id in (
        effective_plan_config.model_id,
        reflection_model_id or settings.agent.reflection.model_id,
    ):
        if model_id and model_id not in agent_model_ids:
            agent_model_ids.append(model_id)
    model_registry = ModelRegistry(
        provider_registry,
        default_model_id=target_model_id,
        model_ids=agent_model_ids,
        fallback_model_ids=settings.llm.fallback_model_ids,
    )
    # 未显式注入时使用内置基础 system prompt；lifespan 会注入带领域信息的版本。
    if system_prompt is None:
        system_prompt = get_default_prompt_registry().render_text(
            PromptId.AGENT_SYSTEM,
            assistant_name=settings.app.name,
            current_date=utcnow().date().isoformat(),
        )
    llm_service = ChatService(
        RoutingChatAdapter(
            model_registry,
            default_model_id=target_model_id,
        ),
        fallback_model_ids=settings.llm.fallback_model_ids,
        enable_fallback=settings.llm.enable_model_fallback,
    )
    # 垂直领域目录：主画像主题 + 子画像细分方向；同时提取领域词用于规则初筛。
    if domain_catalog is None:
        catalog, terms = _load_domain_catalog()
    else:
        catalog, terms = domain_catalog, ()
    planner = query_planner or QueryPlanner(
        effective_plan_config,
        llm=llm_service,
        llm_model_id=target_model_id,
        domain_terms=terms,
    )
    flow = flow_log if flow_log is not None else build_flow_log(settings.app.log)
    return AgentDependencies(
        llm=llm_service,
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
        query_planner=planner,
        domain_catalog=catalog,
        flow_log=flow,
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
        # 轮次召回工具：注册表常驻，运行时按摘要压缩次数追加权限码。
        recall_round_permission=(
            settings.tools.recall_round.permission_code
            if settings.tools.recall_round.enabled
            else None
        ),
        recall_round_min_compressions=(
            settings.tools.recall_round.min_compressions
        ),
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
    resolved_config = config or AgentConfig()
    # history_limit / query_plan 属于依赖端口；未显式传入时跟随 AgentConfig，
    # 避免“AgentLoop 用传入配置、bootstrap 却用全局 Settings”的两套默认值。
    kwargs.setdefault("history_limit", resolved_config.history_limit)
    kwargs.setdefault("query_plan_config", resolved_config.query_plan)
    kwargs.setdefault("reflection_model_id", resolved_config.reflection.model_id)
    dependencies = build_default_agent_dependencies(
        provider_registry=provider_registry,
        **kwargs,
    )
    return await create_agent(resolved_config, dependencies=dependencies)


def _load_domain_catalog() -> tuple[str | None, tuple[str, ...]]:
    """加载主/子画像，生成 Agent 领域目录与规则词表。

    目录文件损坏时只记录 warning，不阻断 Agent 装配；缺失画像时返回空目录，
    查询规划退化为不依赖领域描述的规则/LLM 判断。
    """
    cfg = settings.rag.indexer
    directory = cfg.domain_profile_dir
    if cfg.domain_profile_path:
        path_obj = Path(cfg.domain_profile_path).expanduser()
        if path_obj.is_file() and not directory:
            # path 指向单个文件时，同目录的兄弟画像也按该目录解析。
            directory = str(path_obj.parent)
    names = list_domain_profile_names(
        path=cfg.domain_profile_path,
        directory=directory,
    )
    profiles = []
    terms: list[str] = []
    for name in names:
        try:
            profile = load_domain_profile(
                name,
                path=(
                    cfg.domain_profile_path
                    if cfg.domain_profile_path and Path(cfg.domain_profile_path).stem == name
                    else None
                ),
                directory=directory,
            )
        except Exception:  # noqa: BLE001 - 单个画像失败不影响其它画像
            logger.warning("加载领域画像失败，跳过: %s", name, exc_info=True)
            continue
        profiles.append(profile)
        terms.extend(item.name for item in profile.whitelist)
        for entity in profile.whitelist:
            terms.extend(entity.aliases)
        for entity_type in profile.entity_types:
            if entity_type.label:
                terms.append(entity_type.label)
            terms.extend(entity_type.examples[:4])
    catalog = domain_catalog_text(profiles) or None
    deduped: list[str] = []
    seen: set[str] = set()
    for item in terms:
        value = str(item).strip()
        if len(value) < 2 or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        deduped.append(value)
    return catalog, tuple(deduped)
