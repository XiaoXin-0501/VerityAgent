"""Agent 默认依赖装配（framework）。

把 Provider chat 模型、PG 消息/租约、工具注册表/流水线、ContextAssembler、
记忆画像/队列、RAG 适配器、事实存储、压缩器与事件存储拼成
``AgentDependencies``；调用方负责基础设施的生命周期与 ``registry.init_all()``。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry
from config import settings
from config.agent import AgentConfig
from config.domain import (
    domain_catalog_text,
    domain_entity_type_text,
    list_domain_profile_names,
    load_domain_profile,
)
from prompt import PromptId, get_default_prompt_registry
from utils.time import utcnow
from .adapters.facts import WeaviateFactStore
from .adapters.kb import PgKbCatalogAdapter
from .adapters.llm import RoutingChatAdapter
from .adapters.memory import PgMemoryProfileAdapter
from .adapters.pg import PgLeaseAdapter, PgMemoryQueueAdapter, PgMessageAdapter
from .adapters.rag import RetrieverRagAdapter
from .adapters.tools import PipelineToolAdapter
from .context import ContextAssembler
from .facade import Agent, create_agent
from .llm_service import ChatService, ModelRegistry
from .query_plan import QueryPlanner
from .runtime.background import BackgroundTaskRunner
from .runtime.event_store import RedisEventStore
from .runtime.flow import build_flow_log
from .runtime.ports import AgentDependencies

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
        kb_catalog: Any | None = None,
        compressor: Any | None = None,
        facts: Any | None = None,
        event_store: Any | None = None,
        query_planner: Any | None = None,
        query_plan_config: Any | None = None,
        reflection_model_id: str | None = None,
        flow_log: Any | None = None,
        domain_catalog: str | None = None,
        domain_entity_types: str | None = None,
        domain_noise_terms: tuple[str, ...] | None = None,
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

    # 不能用 ``tool_registry or ...``：ToolRegistry 定义了 __len__，空注册表
    # 会被判为假值，导致调用方显式注入的空注册表被意外替换。
    registry = (
        tool_registry
        if tool_registry is not None
        else build_tool_registry(
            services=services,
            tools_config=settings.tools,
        )
    )
    pipeline = (
        tool_pipeline
        if tool_pipeline is not None
        else build_tool_pipeline(registry, settings.tools)
    )
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
    # 实体类型清单单独传给查询规划 Prompt，避免模板写死某个领域（疾病/药物…）。
    entity_types = ""
    noise_terms: tuple[str, ...] = ()
    if domain_catalog is None:
        catalog, terms, entity_types, noise_terms = _load_domain_catalog()
    else:
        catalog, terms = domain_catalog, ()
    planner = query_planner or QueryPlanner(
        effective_plan_config,
        llm=llm_service,
        llm_model_id=target_model_id,
        domain_terms=terms,
        domain_entity_types=domain_entity_types or entity_types,
        domain_noise_terms=domain_noise_terms or noise_terms,
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
        # 默认用 PG 知识与权限表解析可读知识库；INJECTED 与 search_kb 共用。
        kb_catalog=(
            kb_catalog if kb_catalog is not None else PgKbCatalogAdapter()
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


def _load_domain_catalog() -> tuple[str | None, tuple[str, ...], str, tuple[str, ...]]:
    """加载主/子画像，生成 Agent 领域目录、规则词表、类型参考与噪声词。

    noise_terms 只用于规划实体后过滤；别名与类型/关系黑名单同样不进入
    查询规划 Prompt。目录文件损坏时只记录 warning，不阻断 Agent 装配。
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
        terms.extend(item.name for item in profile.aliases)
        for entity in profile.aliases:
            terms.extend(entity.aliases)
        for entity_type in profile.entity_types:
            if entity_type.label:
                terms.append(entity_type.label)
            terms.extend(entity_type.examples[:4])
    # 配置的 master_profile 排在最前：同一部署可能保留多个参考主画像，
    # 目录/类型参考应优先展示当前领域，避免人类医疗示例挤占宠物医疗示例。
    profiles.sort(
        key=lambda item: (
            item.name != cfg.master_profile,
            not item.is_master,
            item.name,
        )
    )
    catalog = domain_catalog_text(profiles) or None
    entity_types = domain_entity_type_text(profiles)
    noise_terms: set[str] = set()
    for profile in profiles:
        noise_terms.update(profile.noise_keys)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in terms:
        value = str(item).strip()
        if len(value) < 2 or value.casefold() in seen:
            continue
        seen.add(value.casefold())
        deduped.append(value)
    return catalog, tuple(deduped), entity_types, tuple(sorted(noise_terms))
