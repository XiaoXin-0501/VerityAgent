"""应用生命周期：连接池、schema、Provider 注册表、后台资源。

外部存储（MinIO / Weaviate / Neo4j）默认占位关闭；打开后若连接失败，
按各自的 ``required_for_startup`` 决定是 fail-fast 还是降级启动。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import settings
from exception import ConfigurationError, RedisError
from agent.bootstrap import build_default_agent
from agent.adapters.hooks import DispatcherHookAdapter
from agent.hooks.builtin.factory import build_dispatcher
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry
from provider.factory import build_registry
from prompt import build_prompt_registry
from rag.bootstrap import build_default_retriever
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


async def _build_rag_retriever(registry):
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动/关闭全部基础设施；任一步失败也按相反顺序释放已建资源。"""
    registry = None
    hook_dispatcher = None
    agent = None
    rag_retriever = None
    redis_ready = False
    try:
        await init_pg()

        # 首版没有迁移框架：dev/test 幂等建表；prod 只连接，不自动改结构
        if not settings.app.is_prod:
            await init_schema()

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
        app.state.prompt_registry = build_prompt_registry(settings.prompts)
        await registry.init_all()
        app.state.provider_registry = registry
        rag_retriever = None
        try:
            rag_retriever = await _build_rag_retriever(registry)
        except Exception as exc:  # noqa: BLE001 - RAG 装配失败不阻断普通对话
            logger.warning("RAG Retriever 装配失败，降级为无知识库工具: %s", exc)
        app.state.rag_retriever = rag_retriever
        hook_dispatcher = build_dispatcher(
            settings.hooks,
            conversation_repository=ConversationRepository(),
        )
        app.state.hook_dispatcher = hook_dispatcher
        tool_registry = build_tool_registry(retriever=rag_retriever)
        app.state.tool_registry = tool_registry
        tool_pipeline = build_tool_pipeline(tool_registry, settings.tools)
        app.state.tool_pipeline = tool_pipeline
        agent = await build_default_agent(
            provider_registry=registry,
            tool_registry=tool_registry,
            tool_pipeline=tool_pipeline,
            services=app.state,
            hooks=DispatcherHookAdapter(hook_dispatcher),
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
