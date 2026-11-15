"""应用生命周期：连接池、schema、Provider 注册表、后台资源。

外部存储（MinIO / Weaviate / Neo4j）默认占位关闭；打开后若连接失败，
按各自的 ``required_for_startup`` 决定是 fail-fast 还是降级启动。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from exception import ConfigurationError, RedisError
from agent.bootstrap import build_default_agent
from agent.adapters.facts import WeaviateFactStore
from agent.adapters.hooks import DispatcherHookAdapter
from agent.hooks.builtin.factory import build_dispatcher
from agent.recovery import (
    recover_expired_leases,
    recover_missing_tool_messages,
    recover_streaming_messages,
)
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry
from agent.workers import (
    CompressionService,
    ConflictPenaltyQueue,
    FactConfidencePenaltyWorker,
    FactConflictService,
    FactReconciliationWorker,
    MemoryProfileWorker,
    UserDeletionWorker,
)
from prompt import PromptId, build_prompt_registry
from provider.factory import build_registry
from rag.bootstrap import build_default_retriever
from rag.indexer.enrichment.domain import load_domain_profile
from store.minio import close_minio, init_minio
from store.neo4j import (
    check_schema as check_graph_schema,
    close_neo4j,
    get_neo4j,
    init_neo4j,
    init_schema as init_graph_schema,
)
from store.pg import close_pg, init_pg, init_schema
from store.pg.repositories import ConversationRepository
from store.redis import close_redis, init_redis
from store.weaviate import (
    SchemaManager,
    close_weaviate,
    get_weaviate,
    init_weaviate,
)
from utils.time import utcnow

logger = logging.getLogger(__name__)

__all__ = ["lifespan"]


async def _optional_store(name: str, required: bool, action) -> bool:
    """执行外部存储初始化；非 required 时失败只记录并返回 False。"""
    try:
        await action()
        return True
    except Exception as exc:
        if required:
            raise
        logger.warning("%s 初始化失败，按降级模式启动: %s", name, exc)
        return False


def _resolve_model(
    registry,
    model_id: str | None,
    *,
    fallback: str | None = None,
    kind: str | None = None,
):
    """按应用模型 id 解析句柄；缺失/禁用/类型不符时返回 None。"""
    target = model_id or fallback
    if not target:
        return None
    try:
        model = registry.get_model_by_id(target)
    except ConfigurationError as exc:
        logger.warning("RAG 模型不可用: %s (%s)", target, exc)
        return None
    actual = getattr(getattr(model, "spec", None), "kind", None)
    actual_value = getattr(actual, "value", actual)
    if kind is not None and actual_value is not None and actual_value != kind:
        logger.warning("RAG 模型类型不符: %s (期望 %s, 实际 %s)", target, kind, actual_value)
        return None
    return model


def _default_chat_model_id() -> str | None:
    """返回默认 chat 模型 id；未配置时取第一个启用的 chat 模型。"""
    if settings.llm.default_model_id:
        return settings.llm.default_model_id
    return next(
        (
            item.id
            for item in settings.models
            if item.enabled and item.kind.value == "chat"
        ),
        None,
    )


async def _build_rag_retriever(registry, prompt_registry):
    """按 Settings 构造 RAG Retriever；缺少 embedding/向量库时返回 None。"""
    cfg = settings.rag
    if not settings.weaviate.enabled:
        logger.info("Weaviate 未启用，跳过 RAG Retriever 装配")
        return None
    embedding_id = cfg.indexer.embedding_model_id
    if not embedding_id:
        logger.warning("未配置 RAG embedding 模型，跳过 RAG Retriever 装配")
        return None
    try:
        embedding_model = registry.get_model_by_id(embedding_id)
    except ConfigurationError as exc:
        logger.warning("RAG embedding 模型不可用: %s", exc)
        return None
    default_chat = _default_chat_model_id()
    rerank_model = (
        _resolve_model(registry, cfg.retriever.rerank_model_id, kind="rerank")
        if cfg.retriever.rerank_enabled
        else None
    )
    query_rewrite_model = (
        _resolve_model(
            registry,
            cfg.retriever.query_rewrite_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if cfg.retriever.query_rewrite_enabled
        else None
    )
    intent_model = (
        _resolve_model(
            registry,
            cfg.retriever.intent_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if cfg.retriever.intent_llm_enabled
        else None
    )
    retriever = await build_default_retriever(
        cfg.retriever,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        query_rewrite_model=query_rewrite_model,
        intent_model=intent_model,
        prompt_registry=prompt_registry,
        domain_profile_name=cfg.indexer.domain_profile,
        embedding_dimensions=cfg.indexer.embedding_dimensions,
        embedding_version=cfg.indexer.embedding_version,
        max_query_chars=cfg.max_query_chars,
    )
    logger.info(
        "RAG Retriever 就绪: embedding=%s rerank=%s rewrite=%s intent=%s",
        embedding_id,
        cfg.retriever.rerank_model_id,
        cfg.retriever.query_rewrite_enabled,
        cfg.retriever.intent_llm_enabled,
    )
    return retriever


def _build_memory_worker(registry, hook_dispatcher, prompt_registry):
    """按配置构造记忆抽取 worker；未启用/无模型时返回 None。"""
    cfg = settings.memory
    if not cfg.worker_enabled:
        return None
    default_chat = _default_chat_model_id()
    model_id = cfg.extraction_model_id or default_chat
    model = _resolve_model(
        registry,
        model_id,
        fallback=default_chat,
        kind="chat",
    )
    if model is None:
        logger.warning("记忆抽取 worker 缺少可用 chat 模型，跳过启动")
        return None
    conflict_service = None
    if cfg.fact_conflict_enabled:
        conflict_model_id = cfg.fact_conflict_model_id or model_id
        conflict_model = _resolve_model(
            registry,
            conflict_model_id,
            fallback=model_id,
            kind="chat",
        )
        if conflict_model is not None:
            conflict_service = FactConflictService(
                model=conflict_model,
                model_id=str(conflict_model_id),
                max_candidates=cfg.fact_conflict_max_candidates,
                timeout_s=cfg.fact_conflict_timeout_s,
                max_output_tokens=cfg.fact_conflict_max_output_tokens,
                prompts=prompt_registry,
            )
    return MemoryProfileWorker(
        cfg,
        model=model,
        model_id=str(model_id),
        hook_dispatcher=hook_dispatcher,
        facts_store=(
            WeaviateFactStore()
            if cfg.fact_write_enabled and settings.weaviate.enabled
            else None
        ),
        conflict_service=conflict_service,
        prompts=prompt_registry,
    )


def _build_compressor(registry, hook_dispatcher, prompt_registry):
    """按配置构造回合后摘要压缩服务；未启用/无模型时返回 None。"""
    cfg = settings.memory
    if not cfg.compression_enabled:
        return None
    default_chat = _default_chat_model_id()
    model_id = cfg.compression_model_id or default_chat
    model = _resolve_model(
        registry,
        model_id,
        fallback=default_chat,
        kind="chat",
    )
    if model is None:
        logger.warning("会话压缩缺少可用 chat 模型，跳过压缩")
        return None
    return CompressionService(
        cfg,
        model=model,
        model_id=str(model_id),
        hook_dispatcher=hook_dispatcher,
        prompts=prompt_registry,
    )


def _build_deletion_worker(config):
    """按配置构造用户删除 worker；未启用时返回 None。"""
    if not config.deletion_worker_enabled:
        return None
    return UserDeletionWorker(config)


def _build_reconciliation_worker(config):
    """按配置构造事实对账 worker；未启用/无 Weaviate 时返回 None。"""
    if not config.reconciliation_worker_enabled:
        return None
    if not settings.weaviate.enabled:
        logger.warning("Weaviate 未启用，跳过事实对账 worker")
        return None
    return FactReconciliationWorker(
        config,
        facts_store=WeaviateFactStore(),
    )


def _build_penalty_worker(config, queue):
    """按配置构造事实置信度惩罚 worker；未启用/无 Weaviate 时返回 None。"""
    if not config.penalty_worker_enabled:
        return None
    if not settings.weaviate.enabled:
        logger.warning("Weaviate 未启用，跳过事实惩罚 worker")
        return None
    return FactConfidencePenaltyWorker(
        queue=queue,
        facts_store=WeaviateFactStore(),
        disputed_threshold=config.penalty_disputed_threshold,
        batch_size=config.penalty_batch_size,
        interval_s=config.penalty_interval_s,
        max_attempts=config.penalty_max_attempts,
    )


def _build_agent_system_prompt(prompt_registry) -> str:
    """渲染 Agent 基础 system prompt；领域画像不可用时退回通用描述。"""
    domain_name = "通用"
    domain_description = ""
    try:
        profile = load_domain_profile(
            settings.rag.indexer.domain_profile,
            path=settings.rag.indexer.domain_profile_path,
            directory=settings.rag.indexer.domain_profile_dir,
        )
        domain_name = profile.display_name or profile.name
        domain_description = profile.description or ""
    except Exception:  # noqa: BLE001 - 画像缺失不应阻断普通对话
        logger.warning(
            "加载领域画像失败，Agent system prompt 使用通用领域描述",
            exc_info=True,
        )
    return prompt_registry.render_text(
        PromptId.AGENT_SYSTEM,
        assistant_name=settings.app.name,
        domain_name=domain_name,
        domain_description=domain_description,
        current_date=utcnow().date().isoformat(),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动/关闭全部基础设施；任一步失败也按相反顺序释放已建资源。"""
    registry = None
    hook_dispatcher = None
    agent = None
    rag_retriever = None
    memory_worker = None
    memory_worker_task = None
    deletion_worker = None
    deletion_worker_task = None
    reconciliation_worker = None
    reconciliation_worker_task = None
    penalty_worker = None
    penalty_worker_task = None
    compressor = None
    redis_ready = False
    try:
        await init_pg()

        # 首版没有迁移框架：dev/test 幂等建表；prod 只连接，不自动改结构
        if not settings.app.is_prod:
            await init_schema()

        # 进程重启后先收敛过期 run 租约，避免旧 run 阻塞新对话。
        try:
            await recover_expired_leases()
            await recover_streaming_messages()
            await recover_missing_tool_messages()
        except Exception:  # noqa: BLE001 - 恢复失败不阻断启动
            logger.warning("启动期 run/消息恢复失败", exc_info=True)

        try:
            await init_redis()
            redis_ready = True
        except (RedisError, OSError) as exc:
            # 缓存不是权威；Redis 不可用时业务回源，不阻断启动
            logger.warning("Redis 不可用，缓存降级: %s", exc)

        minio_ready = False
        if settings.minio.enabled:
            minio_ready = await _optional_store(
                "MinIO",
                settings.minio.required_for_startup,
                init_minio,
            )

        weaviate_ready = False
        if settings.weaviate.enabled:
            async def _init_weaviate_with_schema() -> None:
                """初始化 Weaviate 并按配置校验/创建 schema。"""
                await init_weaviate()
                if settings.weaviate.startup_schema_check:
                    manager = SchemaManager(settings.weaviate)
                    result = await manager.check_or_ensure(
                        get_weaviate().client,
                        expected_dimensions=settings.rag.indexer.embedding_dimensions,
                    )
                    app.state.weaviate_schema = result.model_dump(mode="json")
                    logger.info(
                        "Weaviate schema 校验通过: %s",
                        result.model_dump(mode="json"),
                    )

            weaviate_ready = await _optional_store(
                "Weaviate",
                settings.weaviate.required_for_startup,
                _init_weaviate_with_schema,
            )

        neo4j_ready = False
        if settings.neo4j.enabled:
            async def _init_neo4j_with_schema() -> None:
                """初始化 Neo4j 并按配置创建与校验图 schema。"""
                await init_neo4j()
                if settings.neo4j.init_schema_on_startup:
                    await init_graph_schema(get_neo4j())
                app.state.neo4j_schema = await check_graph_schema(get_neo4j())

            neo4j_ready = await _optional_store(
                "Neo4j",
                settings.neo4j.required_for_startup,
                _init_neo4j_with_schema,
            )

        registry = build_registry(settings.models)
        # Prompt 模板是启动期资产：目录/manifest/变量不一致直接 fail-fast，
        # 不允许带病启动；运行期注册表只读。
        prompts = build_prompt_registry(settings.prompts)
        app.state.prompt_registry = prompts
        system_prompt = _build_agent_system_prompt(prompts)
        await registry.init_all()
        app.state.provider_registry = registry
        rag_retriever = None
        try:
            rag_retriever = await _build_rag_retriever(registry, prompts)
        except Exception as exc:  # noqa: BLE001 - RAG 装配失败不阻断普通对话
            logger.warning("RAG Retriever 装配失败，降级为无知识库工具: %s", exc)
        app.state.rag_retriever = rag_retriever
        hook_dispatcher = build_dispatcher(
            settings.hooks,
            conversation_repository=ConversationRepository(),
        )
        app.state.hook_dispatcher = hook_dispatcher
        memory_worker = _build_memory_worker(registry, hook_dispatcher, prompts)
        if memory_worker is not None:
            memory_worker_task = asyncio.create_task(
                memory_worker.run_forever(),
                name="verity-memory-worker",
            )
            app.state.memory_worker = memory_worker
        compressor = _build_compressor(registry, hook_dispatcher, prompts)
        deletion_worker = _build_deletion_worker(settings.memory)
        if deletion_worker is not None:
            deletion_worker_task = asyncio.create_task(
                deletion_worker.run_forever(),
                name="verity-deletion-worker",
            )
            app.state.deletion_worker = deletion_worker
        reconciliation_worker = _build_reconciliation_worker(settings.memory)
        if reconciliation_worker is not None:
            reconciliation_worker_task = asyncio.create_task(
                reconciliation_worker.run_forever(),
                name="verity-reconciliation-worker",
            )
            app.state.reconciliation_worker = reconciliation_worker
        penalty_queue = ConflictPenaltyQueue(
            max_len=settings.memory.penalty_queue_max_len
        )
        app.state.conflict_penalty_queue = penalty_queue
        penalty_worker = _build_penalty_worker(settings.memory, penalty_queue)
        if penalty_worker is not None:
            penalty_worker_task = asyncio.create_task(
                penalty_worker.run_forever(),
                name="verity-penalty-worker",
            )
        if (
            settings.rag.retriever.chunk_conflict_tool_enabled
            and not settings.memory.penalty_worker_enabled
        ):
            logger.warning(
                "resolve_chunk_conflict 已启用，但 penalty worker 未启用；"
                "惩罚消息只会在有界队列中等待/被淘汰"
            )
        tool_registry = build_tool_registry(
            retriever=rag_retriever,
            conflict_enabled=settings.rag.retriever.chunk_conflict_tool_enabled,
            search_kb=settings.tools.search_kb,
        )
        app.state.tool_registry = tool_registry
        tool_pipeline = build_tool_pipeline(tool_registry, settings.tools)
        app.state.tool_pipeline = tool_pipeline
        agent = await build_default_agent(
            provider_registry=registry,
            tool_registry=tool_registry,
            tool_pipeline=tool_pipeline,
            services=app.state,
            hooks=DispatcherHookAdapter(hook_dispatcher),
            rag_retriever=rag_retriever,
            compressor=compressor,
            system_prompt=system_prompt,
        )
        app.state.agent = agent
        app.state.redis_ready = redis_ready
        app.state.minio_ready = minio_ready
        app.state.weaviate_ready = weaviate_ready
        app.state.neo4j_ready = neo4j_ready

        yield
    finally:
        if agent is not None:
            await agent.close()
        if memory_worker is not None:
            memory_worker.stop()
        if memory_worker_task is not None:
            memory_worker_task.cancel()
            await asyncio.gather(memory_worker_task, return_exceptions=True)
        if deletion_worker is not None:
            deletion_worker.stop()
        if deletion_worker_task is not None:
            deletion_worker_task.cancel()
            await asyncio.gather(deletion_worker_task, return_exceptions=True)
        if reconciliation_worker is not None:
            reconciliation_worker.stop()
        if reconciliation_worker_task is not None:
            reconciliation_worker_task.cancel()
            await asyncio.gather(
                reconciliation_worker_task,
                return_exceptions=True,
            )
        if penalty_worker is not None:
            penalty_worker.stop()
        if penalty_worker_task is not None:
            penalty_worker_task.cancel()
            await asyncio.gather(
                penalty_worker_task,
                return_exceptions=True,
            )
        if rag_retriever is not None:
            await rag_retriever.close()
        if registry is not None:
            await registry.close_all()
        if hook_dispatcher is not None:
            await hook_dispatcher.close()
        # close_* 都是幂等的；即使初始化中途失败也要释放已建立的连接
        await close_neo4j()
        await close_weaviate()
        await close_minio()
        if redis_ready:
            await close_redis()
        await close_pg()
