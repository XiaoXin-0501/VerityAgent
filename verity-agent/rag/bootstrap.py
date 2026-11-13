"""RAG 依赖装配入口。

门面只接收 ``RagDependencies``；默认实现（Weaviate/Neo4j/MinIO/PG）在这里
集中接线，避免 facade 反向依赖具体存储。
"""

from __future__ import annotations

import logging
from typing import Any

from config.rag import IndexerConfig
from prompt import PromptRegistry

from .facade import Indexer, RagDependencies, create_indexer
from .indexer import (
    ChatCodeDescriber,
    ChatTableDescriber,
    DefaultIndexingPipeline,
    EmbeddingService,
    GraphExtractor,
    IndexerPolicy,
    InMemoryJobStore,
    IndexingRepository,
    JobStore,
    MinioManifestStore,
    MinioSourceStore,
    Neo4jGraphIndexStore,
    PgIndexingRepository,
    TextIngestionPipeline,
    WeaviateChunkStore,
    build_graph_extractor,
    get_indexer_policy,
)

__all__ = [
    "build_dependencies",
    "build_default_indexing_pipeline",
    "build_default_indexer",
    "build_table_describer_from_registry",
    "ensure_default_vector_schema",
]

logger = logging.getLogger(__name__)


def build_dependencies(
    *,
    indexing_pipeline: Any | None = None,
    retrieval_pipeline: Any | None = None,
    embedding_model: Any | None = None,
    rerank_model: Any | None = None,
    vector_store: Any | None = None,
    graph_store: Any | None = None,
    manifest_store: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> RagDependencies:
    """构造可注入的依赖容器；不主动连接任何外部服务。"""
    return RagDependencies(
        indexing_pipeline=indexing_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        vector_store=vector_store,
        graph_store=graph_store,
        manifest_store=manifest_store,
        metadata=dict(metadata or {}),
    )


def build_default_indexing_pipeline(
    config: IndexerConfig | None = None,
    *,
    embedding_model: Any,
    vector_store: WeaviateChunkStore | None = None,
    graph_store: Neo4jGraphIndexStore | None = None,
    manifest_store: MinioManifestStore | None = None,
    source_store: MinioSourceStore | None = None,
    repository: IndexingRepository | None = None,
    job_store: JobStore | None = None,
    graph_extractor: GraphExtractor | None = None,
    graph_chat_model: Any | None = None,
    prompt_registry: PromptRegistry | None = None,
    indexer_policy: IndexerPolicy | None = None,
    table_summary_chat_model: Any | None = None,
    code_summary_chat_model: Any | None = None,
    table_describer: Any | None = None,
    code_describer: Any | None = None,
    description_refiner: Any | None = None,
    worker_id: str = "rag-indexer",
    run_in_background: bool = True,
) -> DefaultIndexingPipeline:
    """装配默认 v1 入库流水线。

    只构造对象，不建立连接；首次写 Weaviate/Neo4j/MinIO/PG 时才真正连接。
    ``vector_store.ensure_schema()`` 应在应用启动时单独调用（dev/test 可
    自动建 collection，prod 只校验）。

    图抽取由 ``graph_extractor`` 显式注入时优先使用；否则按
    ``IndexerConfig.graph_extraction_mode`` 构建。``graph_chat_model`` 只在
    ``llm/hybrid`` 模式下使用，未注入时自动降级为领域词典。
    ``prompt_registry`` 用于外部覆盖 Prompt 模板；为空时使用内置模板。
    ``indexer_policy`` 用于覆盖质量扣分/描述标题/阶段进度；为空时使用内置
    ``config/rag_policy.json``。

    表格/代码语义描述：显式注入 ``table_describer`` / ``code_describer`` 优先；
    否则在对应 LLM 开关打开且注入 chat model 时构造 ChatDescriber，用于
    “heading + 描述”的向量化文本。
    """
    cfg = config or IndexerConfig()
    policy = indexer_policy or get_indexer_policy(
        cfg.indexer_policy_path
    )
    prompts = prompt_registry
    table_describer = table_describer or _build_table_describer(
        cfg, prompts, table_summary_chat_model
    )
    code_describer = code_describer or _build_code_describer(
        cfg, prompts, code_summary_chat_model
    )
    return DefaultIndexingPipeline(
        cfg,
        ingestion=TextIngestionPipeline(
            cfg,
            description_refiner=description_refiner,
            policy=policy,
            table_describer=table_describer,
            code_describer=code_describer,
        ),
        embedding=EmbeddingService(
            embedding_model,
            model_id=cfg.embedding_model_id,
            batch_size=cfg.embedding_batch_size,
        ),
        vector_store=vector_store or WeaviateChunkStore(),
        graph_store=graph_store or Neo4jGraphIndexStore(),
        manifest_store=manifest_store or MinioManifestStore(),
        source_store=source_store or MinioSourceStore(),
        repository=repository or PgIndexingRepository(),
        job_store=job_store or InMemoryJobStore(),
        graph_extractor=graph_extractor
        or build_graph_extractor(
            cfg,
            chat_model=graph_chat_model,
            prompts=prompt_registry,
            indexer_policy=policy,
        ),
        policy=policy,
        worker_id=worker_id,
        run_in_background=run_in_background,
    )


async def build_default_indexer(
    config: IndexerConfig | None = None,
    *,
    embedding_model: Any,
    pipeline: DefaultIndexingPipeline | None = None,
    **kwargs: Any,
) -> Indexer:
    """创建使用默认流水线的 ``Indexer``；调用方负责 ``close()``。"""
    cfg = config or IndexerConfig()
    default_pipeline = pipeline or build_default_indexing_pipeline(
        cfg,
        embedding_model=embedding_model,
        **kwargs,
    )
    return await create_indexer(
        cfg,
        dependencies=build_dependencies(
            indexing_pipeline=default_pipeline,
            embedding_model=embedding_model,
            vector_store=kwargs.get("vector_store"),
            graph_store=kwargs.get("graph_store"),
            manifest_store=kwargs.get("manifest_store"),
        ),
    )


async def ensure_default_vector_schema(
    config: IndexerConfig | None = None,
    *,
    vector_store: WeaviateChunkStore | None = None,
):
    """启动期校验/创建 DocChunk schema（D-97）；返回 SchemaCheckResult。"""
    cfg = config or IndexerConfig()
    store = vector_store or WeaviateChunkStore()
    return await store.ensure_schema(
        expected_dimensions=cfg.embedding_dimensions,
    )


def build_table_describer_from_registry(
    registry: Any,
    config: IndexerConfig,
    *,
    prompts: PromptRegistry | None = None,
):
    """从 ProviderRegistry 解析 ``table_summary_model_id`` 并构造描述器。

    生产应用或测试脚本可以复用这个入口，避免自己拼 ChatModel/Describer；
    registry 需先 ``init_all()``。未启用或未配置模型时返回 None。
    """
    if not config.table_summary_llm_enabled or not config.table_summary_model_id:
        return None
    chat_model = registry.get_model_by_id(config.table_summary_model_id)
    return _build_table_describer(config, prompts, chat_model)


def _build_table_describer(
    config: IndexerConfig,
    prompts: PromptRegistry | None,
    chat_model: Any | None,
):
    """按配置构造表格/Excel 语义描述器；未启用或未注入模型返回 None。"""
    if not config.table_summary_llm_enabled:
        return None
    if chat_model is None:
        logger.warning(
            "table_summary_llm_enabled=true 但未注入 table chat model，"
            "表格/Excel 描述将只使用本地 context/caption/summary"
        )
        return None
    model_id = (
        config.table_summary_model_id
        or getattr(getattr(chat_model, "spec", None), "id", None)
        or "table-describer"
    )
    return ChatTableDescriber(
        chat_model,
        model_id=model_id,
        prompts=prompts,
        temperature=config.semantic_description_temperature,
        max_output_tokens=config.semantic_description_max_output_tokens,
        max_chars=config.semantic_description_max_chars,
    )


def _build_code_describer(
    config: IndexerConfig,
    prompts: PromptRegistry | None,
    chat_model: Any | None,
):
    """按配置构造代码语义描述器；未启用或未注入模型返回 None。"""
    if not config.code_summary_enabled:
        return None
    if chat_model is None:
        logger.warning(
            "code_summary_enabled=true 但未注入 code chat model，"
            "代码描述将只使用本地 context/docstring"
        )
        return None
    model_id = (
        config.code_summary_model_id
        or getattr(getattr(chat_model, "spec", None), "id", None)
        or "code-describer"
    )
    return ChatCodeDescriber(
        chat_model,
        model_id=model_id,
        prompts=prompts,
        temperature=config.semantic_description_temperature,
        max_output_tokens=config.semantic_description_max_output_tokens,
        max_chars=config.semantic_description_max_chars,
    )
