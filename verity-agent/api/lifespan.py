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
from exception import RedisError
from agent.hooks.builtin.factory import build_dispatcher
from agent.tools.bootstrap import build_tool_pipeline, build_tool_registry
from provider.factory import build_registry
from prompt import build_prompt_registry
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_pg()

    # 首版没有迁移框架：dev/test 幂等建表；prod 只连接，不自动改结构
    if not settings.app.is_prod:
        await init_schema()

    redis_ready = False
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
                logger.info("Weaviate schema 校验通过: %s", result.model_dump(mode="json"))

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
    hook_dispatcher = build_dispatcher(
        settings.hooks,
        conversation_repository=ConversationRepository(),
    )
    app.state.hook_dispatcher = hook_dispatcher
    tool_registry = build_tool_registry()
    app.state.tool_registry = tool_registry
    app.state.tool_pipeline = build_tool_pipeline(tool_registry, settings.tools)
    app.state.redis_ready = redis_ready
    app.state.minio_ready = minio_ready
    app.state.weaviate_ready = weaviate_ready
    app.state.neo4j_ready = neo4j_ready

    try:
        yield
    finally:
        await registry.close_all()
        await hook_dispatcher.close()
        # close_* 都是幂等的；即使初始化中途失败也要释放已建立的连接
        await close_neo4j()
        await close_weaviate()
        await close_minio()
        if redis_ready:
            await close_redis()
        await close_pg()
