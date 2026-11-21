"""应用生命周期：连接池、schema、Provider 注册表、后台资源。

外部存储（MinIO / Weaviate / Neo4j）默认占位关闭；打开后若连接失败，
按各自的 ``required_for_startup`` 决定是 fail-fast 还是降级启动。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from agent.adapters.facts import WeaviateFactStore
from agent.adapters.hooks import DispatcherHookAdapter
from agent.adapters.kb import PgKbCatalogAdapter
from agent.adapters.rag_events import HookRagEventSink
from agent.bootstrap import build_default_agent
from agent.hooks.builtin.factory import build_dispatcher
from agent.recovery import (
    recover_expired_leases,
    recover_missing_tool_messages,
    recover_streaming_messages,
)
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry
from agent.workers import (
    CompressionService,
    FactReconciliationWorker,
    MemoryProfileWorker,
    UserDeletionWorker,
)
from config import settings
from config.domain import load_domain_profile, load_master_profile
from exception import ConfigurationError, RedisError
from prompt import PromptId, build_prompt_registry
from provider.factory import build_registry
from rag.bootstrap import build_default_indexer, build_default_retriever
from rag.retriever.chunk_conflict import ChunkConflictService
from store.minio import close_minio, init_minio
from store.neo4j import (
    check_schema as check_graph_schema,
    close_neo4j,
    get_neo4j,
    init_neo4j,
    init_schema as init_graph_schema,
)
from store.pg import close_pg, init_pg, init_schema
from store.pg.repositories import (
    ConversationMessageRepository,
    ConversationRepository,
    DocumentRepository,
    PermissionRepository,
)
from store.redis import close_redis, init_redis
from store.weaviate import (
    SchemaManager,
    close_weaviate,
    get_weaviate,
    init_weaviate,
)
from utils.time import utcnow
from utils.tokens import configure_tokenizer
from .logging_setup import setup_logging, shutdown_logging

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


def _load_domain_profile_by_name(name: str):
    """按 KB 绑定的画像名加载画像；全局 path 仅匹配同名画像。"""
    configured_path = settings.rag.indexer.domain_profile_path
    directory = settings.rag.indexer.domain_profile_dir
    if configured_path:
        path_obj = Path(configured_path).expanduser()
        if path_obj.is_file():
            if path_obj.stem == name:
                return load_domain_profile(name, path=path_obj)
            # 自定义目录里的其它子画像仍应与该文件同目录查找。
            directory = directory or str(path_obj.parent)
    return load_domain_profile(
        name,
        directory=directory,
    )


async def _build_rag_retriever(
        registry,
        prompt_registry,
        *,
        available: bool = True,
        event_sink=None,
):
    """按 Settings 构造 RAG Retriever；向量库未就绪时返回 None。"""
    cfg = settings.rag
    if not settings.weaviate.enabled:
        logger.info("Weaviate 未启用，跳过 RAG Retriever 装配")
        return None
    if not available:
        logger.warning("Weaviate 未就绪，跳过 RAG Retriever 装配")
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
    rerank_model = (
        _resolve_model(registry, cfg.retriever.rerank_model_id, kind="rerank")
        if cfg.retriever.rerank_enabled
        else None
    )
    retriever = await build_default_retriever(
        cfg.retriever,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        # 显式加载，确保自定义 domain_profile_dir/path 生效；仅传名字会让
        # rag.bootstrap 回退内置目录。
        domain_profile=_load_domain_profile_by_name(cfg.indexer.domain_profile),
        embedding_dimensions=cfg.indexer.embedding_dimensions,
        embedding_version=cfg.indexer.embedding_version,
        max_query_chars=cfg.max_query_chars,
        domain_profile_resolver=_load_domain_profile_by_name,
        event_sink=event_sink,
    )
    logger.info(
        "RAG Retriever 就绪: embedding=%s rerank=%s",
        embedding_id,
        cfg.retriever.rerank_model_id,
    )
    return retriever


async def _build_rag_indexer(
        registry,
        prompt_registry,
        *,
        available: bool = True,
        event_sink=None,
):
    """按 Settings 构造 RAG Indexer；MinIO/Weaviate/embedding 未就绪时返回 None。"""
    cfg = settings.rag
    if not settings.weaviate.enabled or not settings.minio.enabled:
        logger.info("Weaviate/MinIO 未启用，跳过 RAG Indexer 装配")
        return None
    if not available:
        logger.warning("MinIO/Weaviate 未就绪，跳过 RAG Indexer 装配")
        return None
    embedding_id = cfg.indexer.embedding_model_id
    if not embedding_id:
        logger.warning("未配置 RAG embedding 模型，跳过 RAG Indexer 装配")
        return None
    try:
        embedding_model = registry.get_model_by_id(embedding_id)
    except ConfigurationError as exc:
        logger.warning("RAG embedding 模型不可用: %s", exc)
        return None
    # 入库阶段的描述/表格/代码/图抽取模型都按配置解析；配置了模型 id 就注入，
    # 否则显式降级，避免“配置打开了但实际一直没生效”的静默偏差。
    default_chat = _default_chat_model_id()
    indexer_cfg = cfg.indexer
    description_model = (
        _resolve_model(
            registry,
            indexer_cfg.description_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if indexer_cfg.description_llm_enabled
        else None
    )
    table_model = (
        _resolve_model(
            registry,
            indexer_cfg.table_summary_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if indexer_cfg.table_summary_llm_enabled
        else None
    )
    code_model = (
        _resolve_model(
            registry,
            indexer_cfg.code_summary_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if indexer_cfg.code_summary_enabled
        else None
    )
    graph_model = (
        _resolve_model(
            registry,
            indexer_cfg.graph_model_id,
            fallback=default_chat,
            kind="chat",
        )
        if indexer_cfg.graph_build_enabled
        else None
    )
    if indexer_cfg.graph_build_enabled and graph_model is None:
        logger.warning(
            "图构建已开启但未解析到 graph chat model；"
            "入库时图抽取会因缺少 LLM 失败"
        )
    logger.info(
        "RAG Indexer 模型: embedding=%s description=%s table=%s code=%s graph=%s",
        embedding_id,
        getattr(getattr(description_model, "spec", None), "id", None),
        getattr(getattr(table_model, "spec", None), "id", None),
        getattr(getattr(code_model, "spec", None), "id", None),
        getattr(getattr(graph_model, "spec", None), "id", None),
    )
    return await build_default_indexer(
        indexer_cfg,
        embedding_model=embedding_model,
        description_chat_model=description_model,
        table_summary_chat_model=table_model,
        code_summary_chat_model=code_model,
        graph_chat_model=graph_model,
        prompt_registry=prompt_registry,
        event_sink=event_sink,
    )


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


def _build_chunk_conflict_service(
    registry,
    prompt_registry,
    *,
    available: bool,
    kb_catalog=None,
):
    """构造知识库片段冲突处理服务；未启用或 Weaviate 不可用时返回 None。"""
    cfg = settings.rag.retriever
    if not cfg.chunk_conflict_tool_enabled:
        return None
    if not available or not settings.weaviate.enabled:
        logger.warning("Weaviate 未就绪，跳过 chunk 冲突处理服务")
        return None
    default_chat = _default_chat_model_id()
    model_id = cfg.chunk_conflict_model_id or default_chat
    model = _resolve_model(
        registry,
        model_id,
        fallback=default_chat,
        kind="chat",
    )
    if model is None:
        logger.warning("chunk 冲突处理缺少可用 chat 模型，跳过")
        return None
    return ChunkConflictService(
        model=model,
        model_id=str(model_id),
        documents=DocumentRepository(),
        timeout_s=cfg.chunk_conflict_timeout_s,
        max_output_tokens=cfg.chunk_conflict_max_output_tokens,
        neighbor_window=cfg.chunk_conflict_neighbor_window,
        prompts=prompt_registry,
        # 冲突工具按 chunk_id 直接读取片段，必须复用与检索一致的 KB 权限
        # 事实源；只凭 kb:read:<id> 会漏掉自定义 required_permission_code。
        kb_catalog=kb_catalog,
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


def _build_agent_system_prompt(prompt_registry) -> str:
    """渲染 Agent 基础 system prompt；使用主画像奠定垂直领域主题。"""
    domain_name = "通用"
    domain_description = ""
    try:
        profile = load_master_profile(
            settings.rag.indexer.master_profile,
            path=(
                settings.rag.indexer.domain_profile_path
                if settings.rag.indexer.domain_profile_path
                   and Path(settings.rag.indexer.domain_profile_path).stem
                   == settings.rag.indexer.master_profile
                else None
            ),
            directory=settings.rag.indexer.domain_profile_dir,
        )
        domain_name = profile.display_name or profile.name
        domain_description = profile.description or ""
    except Exception:  # noqa: BLE001 - 画像缺失不应阻断普通对话
        logger.warning(
            "加载主领域画像失败，Agent system prompt 使用通用领域描述",
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
    # 统一 tokenizer，确保预算/切片/压缩使用同一编码器。
    configure_tokenizer(settings.llm.tokenizer.encoding)
    # 外部 uvicorn 直接加载 api.app:app 时也在这里安装文件日志 handler。
    try:
        app.state.log_file = setup_logging()
    except Exception:  # noqa: BLE001 - 日志初始化失败不阻断启动
        app.state.log_file = None
        logger.warning("文件日志初始化失败，仅使用控制台", exc_info=True)
    if app.state.log_file is not None:
        log_dir = app.state.log_file.parent
        logger.info(
            "日志文件: app=%s access=%s error=%s rag=%s flow=%s",
            app.state.log_file,
            log_dir / settings.app.log.access_file_name,
            log_dir / settings.app.log.error_file_name,
            log_dir / settings.app.log.rag_file_name,
            log_dir / settings.app.log.flow_file_name,
        )
    registry = None
    hook_dispatcher = None
    agent = None
    rag_retriever = None
    rag_indexer = None
    memory_worker = None
    memory_worker_task = None
    deletion_worker = None
    deletion_worker_task = None
    reconciliation_worker = None
    reconciliation_worker_task = None
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
        # 知识库可读目录：INJECTED RAG 与 search_kb 共用，权限判断只查 PG。
        kb_catalog = PgKbCatalogAdapter()
        app.state.kb_catalog = kb_catalog
        # Hooks 先于 RAG 装配：RAG 只向自己的 RagEventSink 发事件，
        # Agent 侧适配器再转成 HookEvent，避免 rag -> agent 反向依赖。
        hook_dispatcher = build_dispatcher(
            settings.hooks,
            conversation_repository=ConversationRepository(),
        )
        app.state.hook_dispatcher = hook_dispatcher
        rag_event_sink = HookRagEventSink(hook_dispatcher)
        rag_retriever = None
        try:
            rag_retriever = await _build_rag_retriever(
                registry,
                prompts,
                available=weaviate_ready,
                event_sink=rag_event_sink,
            )
        except Exception as exc:  # noqa: BLE001 - RAG 装配失败不阻断普通对话
            logger.warning("RAG Retriever 装配失败，降级为无知识库工具: %s", exc)
        app.state.rag_retriever = rag_retriever
        try:
            rag_indexer = await _build_rag_indexer(
                registry,
                prompts,
                available=weaviate_ready and minio_ready,
                event_sink=rag_event_sink,
            )
        except Exception as exc:  # noqa: BLE001 - 管理面入库失败不阻断对话
            logger.warning("RAG Indexer 装配失败，知识库管理不可用: %s", exc)
        app.state.indexer = rag_indexer
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
        # 片段冲突处理服务由工具按需调用，不注册独立 worker。
        chunk_conflict_service = _build_chunk_conflict_service(
            registry,
            prompts,
            available=weaviate_ready,
            kb_catalog=kb_catalog,
        )
        app.state.chunk_conflict_service = chunk_conflict_service
        # 启动期权限定义用于工具注册前过滤；请求期仍按用户实际权限过滤。
        try:
            registered_permissions = (
                await PermissionRepository().list_all_codes()
            )
        except Exception:  # noqa: BLE001 - 权限表不可读时保留全部工具
            logger.warning("读取权限码失败，跳过工具注册权限过滤", exc_info=True)
            registered_permissions = None
        # 轮次召回等工具通过 app.state 共享 PG 仓库；先挂载再装配注册表，
        # 保证 required_services 检查与运行期读取看到同一个服务实例。
        app.state.conversation_messages = ConversationMessageRepository()
        tool_registry = build_tool_registry(
            # 通用依赖包：工厂自行按 key 取 retriever/chunk_conflict_service 等；
            # 新增工具不再需要修改 bootstrap 签名。
            services=app.state,
            available_permissions=registered_permissions,
            tools_config=settings.tools,
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
            kb_catalog=kb_catalog,
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
        if rag_retriever is not None:
            await rag_retriever.close()
        if rag_indexer is not None:
            await rag_indexer.close()
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
        # 最后刷新并关闭文件 handler，避免进程退出时丢失尾部日志。
        shutdown_logging()
