"""RAG 依赖装配入口。

门面只接收 ``RagDependencies``；默认实现（Weaviate/Neo4j/MinIO/PG）在这里
集中接线，避免 facade 反向依赖具体存储。
"""

from __future__ import annotations

import logging
from typing import Any

from config.rag import IndexerConfig, RetrieverConfig
from prompt import PromptRegistry
from store.neo4j import GraphStore
from store.weaviate import WeaviateStore

from .facade import Indexer, RagDependencies, Retriever, create_indexer, create_retriever
from .indexer import (
    ChatCodeDescriber,
    ChatDescriptionRefiner,
    ChatTableDescriber,
    ChunkCache,
    DefaultIndexingPipeline,
    EmbeddingService,
    GraphExtractor,
    InMemoryJobStore,
    IndexingRepository,
    JobStore,
    MinioManifestStore,
    MinioSourceStore,
    Neo4jGraphIndexStore,
    PgIndexingRepository,
    TextIngestionPipeline,
    WeaviateChunkStore,
    build_default_chunk_cache,
    build_graph_extractor,
    load_domain_profile,
)
from .retriever.cache import RetrievalCache
from .retriever.default import DefaultRetrievalPipeline
from .retriever.graph_neo4j import Neo4jGraphExpander
from .retriever.metadata import PgRetrievalMetadataStore, RetrievalMetadataStore
from .retriever.query_llm import ChatIntentClassifier, ChatQueryRewriter
from .retriever.rerank_provider import ProviderReranker
from .retriever.weaviate import WeaviateHybridRecaller, WeaviateNeighborExpander

__all__ = [
    "build_dependencies",
    "build_default_indexing_pipeline",
    "build_default_indexer",
    "build_default_retrieval_pipeline",
    "build_default_retriever",
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
    table_summary_chat_model: Any | None = None,
    code_summary_chat_model: Any | None = None,
    description_chat_model: Any | None = None,
    table_describer: Any | None = None,
    code_describer: Any | None = None,
    description_refiner: Any | None = None,
    cache: ChunkCache | None = None,
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

    表格/代码语义描述：显式注入 ``table_describer`` / ``code_describer`` 优先；
    否则在对应 LLM 开关打开且注入 chat model 时构造 ChatDescriber，用于
    “heading + 描述”的向量化文本。``cache`` 为空时默认使用带熔断的 Redis
    缓存，用于复用 chunk 向量与 LLM 描述；测试可注入 InMemoryChunkCache。

    文档级描述：``description_refiner`` 显式注入优先；否则在
    ``description_llm_enabled`` 且注入 ``description_chat_model`` 时构造
    ``ChatDescriptionRefiner``，用于压缩“标题+长摘要/前言”或精炼弱候选。
    """
    cfg = config or IndexerConfig()
    prompts = prompt_registry
    # 默认使用带熔断的 Redis 缓存；Redis 不可用时自动降级为 miss，不阻塞入库。
    rag_cache = cache
    if rag_cache is None and (
        cfg.embedding_cache_enabled or cfg.description_cache_enabled
    ):
        rag_cache = build_default_chunk_cache()
    table_describer = table_describer or _build_table_describer(
        cfg, prompts, table_summary_chat_model, rag_cache
    )
    code_describer = code_describer or _build_code_describer(
        cfg, prompts, code_summary_chat_model, rag_cache
    )
    description_refiner = description_refiner or _build_description_refiner(
        cfg, prompts, description_chat_model, rag_cache
    )
    return DefaultIndexingPipeline(
        cfg,
        ingestion=TextIngestionPipeline(
            cfg,
            description_refiner=description_refiner,
            table_describer=table_describer,
            code_describer=code_describer,
        ),
        embedding=EmbeddingService(
            embedding_model,
            model_id=cfg.embedding_model_id,
            batch_size=cfg.embedding_batch_size,
            semantic_embedding_enabled=cfg.semantic_embedding_enabled,
            cache=rag_cache,
            cache_enabled=cfg.embedding_cache_enabled,
            cache_ttl_s=cfg.embedding_cache_ttl_s,
            cache_version=cfg.embedding_version,
            configured_dimensions=cfg.embedding_dimensions,
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
        ),
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


def build_default_retrieval_pipeline(
    config: RetrieverConfig | None = None,
    *,
    embedding_model: Any,
    rerank_model: Any | None = None,
    weaviate_store: WeaviateStore | None = None,
    graph_store: GraphStore | None = None,
    metadata_store: RetrievalMetadataStore | None = None,
    cache: RetrievalCache | None = None,
    domain_profile: Any | None = None,
    domain_profile_name: str | None = None,
    embedding_dimensions: int | None = None,
    embedding_version: str = "v1",
    query_rewrite_model: Any | None = None,
    intent_model: Any | None = None,
    prompt_registry: PromptRegistry | None = None,
    max_query_chars: int = 2000,
) -> DefaultRetrievalPipeline:
    """装配默认检索流水线。

    只构造对象，不建立连接。``embedding_model`` 必填；``rerank_model`` 为空
    时流水线自动降级为融合分，并写 ``RERANK_DISABLED`` 告警。

    - ``weaviate_store`` / ``graph_store`` 不传时使用进程默认连接管理器；
    - ``metadata_store`` 默认走 PG，批量查文档状态/描述与 KB 索引版本；
    - ``cache`` 默认使用带熔断的 Redis 缓存；Redis 不可用时降级为 miss；
    - ``embedding_version`` 应传入库时 ``IndexerConfig.embedding_version``，
      用于 query embedding 缓存失效；
    - ``domain_profile`` 为空且 ``domain_profile_name`` 非空时加载对应领域
      画像，用于 query → 图 seed 实体链接；图扩展默认关闭，不额外查询图；
    - ``prompt_registry`` 用于 query 改写/意图分类模板的外部覆盖。
    """
    cfg = config or RetrieverConfig()
    if cache is None and cfg.cache_enabled:
        cache = build_default_chunk_cache()
    embedding = EmbeddingService(
        embedding_model,
        batch_size=1,
        cache=cache,
        cache_enabled=cfg.cache_enabled,
        cache_ttl_s=cfg.query_embedding_cache_ttl_s,
        cache_version=embedding_version,
        configured_dimensions=embedding_dimensions,
    )
    recaller = WeaviateHybridRecaller(
        embedding_service=embedding,
        vector_store=weaviate_store or WeaviateStore(),
        dimensions=embedding_dimensions,
        embedding_version=embedding_version,
    )
    reranker = None
    if cfg.rerank_enabled and rerank_model is not None:
        reranker = ProviderReranker(
            rerank_model,
            model_id=cfg.rerank_model_id,
        )
    profile = domain_profile
    if profile is None and domain_profile_name:
        profile = load_domain_profile(domain_profile_name)
    query_rewriter = None
    if cfg.query_rewrite_enabled and query_rewrite_model is not None:
        query_rewriter = ChatQueryRewriter(
            query_rewrite_model,
            model_id=(
                cfg.query_rewrite_model_id
                or getattr(getattr(query_rewrite_model, "spec", None), "id", "rewrite")
            ),
            max_variants=cfg.query_rewrite_max_variants,
            timeout_s=cfg.query_rewrite_timeout_s,
            cache=cache,
            prompts=prompt_registry,
        )
    intent_classifier = None
    if cfg.intent_llm_enabled and intent_model is not None:
        intent_classifier = ChatIntentClassifier(
            intent_model,
            model_id=(
                cfg.intent_model_id
                or getattr(getattr(intent_model, "spec", None), "id", "intent")
            ),
            timeout_s=cfg.intent_timeout_s,
            cache=cache,
            prompts=prompt_registry,
        )
    return DefaultRetrievalPipeline(
        cfg,
        recaller=recaller,
        reranker=reranker,
        graph_expander=Neo4jGraphExpander(
            graph_store=graph_store or GraphStore()
        ),
        neighbor_expander=WeaviateNeighborExpander(
            vector_store=weaviate_store or WeaviateStore()
        ),
        metadata_store=metadata_store or PgRetrievalMetadataStore(),
        cache=cache,
        domain_profile=profile,
        query_rewriter=query_rewriter,
        intent_classifier=intent_classifier,
        max_query_chars=max_query_chars,
    )


async def build_default_retriever(
    config: RetrieverConfig | None = None,
    *,
    embedding_model: Any,
    rerank_model: Any | None = None,
    pipeline: DefaultRetrievalPipeline | None = None,
    **kwargs: Any,
) -> Retriever:
    """创建使用默认检索流水线的 ``Retriever``；调用方负责 ``close()``。"""
    cfg = config or RetrieverConfig()
    default_pipeline = pipeline or build_default_retrieval_pipeline(
        cfg,
        embedding_model=embedding_model,
        rerank_model=rerank_model,
        **kwargs,
    )
    return await create_retriever(
        cfg,
        dependencies=build_dependencies(
            retrieval_pipeline=default_pipeline,
            embedding_model=embedding_model,
            rerank_model=rerank_model,
            vector_store=kwargs.get("weaviate_store"),
            graph_store=kwargs.get("graph_store"),
        ),
    )


def build_table_describer_from_registry(
    registry: Any,
    config: IndexerConfig,
    *,
    prompts: PromptRegistry | None = None,
    cache: ChunkCache | None = None,
):
    """从 ProviderRegistry 解析 ``table_summary_model_id`` 并构造描述器。

    生产应用或测试脚本可以复用这个入口，避免自己拼 ChatModel/Describer；
    registry 需先 ``init_all()``。未启用或未配置模型时返回 None。
    """
    if not config.table_summary_llm_enabled or not config.table_summary_model_id:
        return None
    chat_model = registry.get_model_by_id(config.table_summary_model_id)
    return _build_table_describer(config, prompts, chat_model, cache)


def _build_description_refiner(
    config: IndexerConfig,
    prompts: PromptRegistry | None,
    chat_model: Any | None,
    cache: ChunkCache | None = None,
):
    """按配置构造文档描述精炼器；未启用或未注入模型返回 None。"""
    if not config.description_llm_enabled:
        return None
    if chat_model is None:
        if config.description_model_id:
            logger.warning(
                "description_llm_enabled=true 但未注入 description chat model，"
                "长摘要/前言将按句截断兜底"
            )
        return None
    model_id = (
        config.description_model_id
        or getattr(getattr(chat_model, "spec", None), "id", None)
        or "description"
    )
    return ChatDescriptionRefiner(
        chat_model,
        model_id=model_id,
        temperature=0.1,
        # 500 字中文描述约需 500+ token，256 会截断；512 留出余量。
        max_output_tokens=512,
        prompts=prompts,
        cache=cache,
        cache_enabled=config.description_cache_enabled,
        cache_ttl_s=config.description_cache_ttl_s,
    )


def _build_table_describer(
    config: IndexerConfig,
    prompts: PromptRegistry | None,
    chat_model: Any | None,
    cache: ChunkCache | None = None,
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
        cache=cache,
        cache_enabled=config.description_cache_enabled,
        cache_ttl_s=config.description_cache_ttl_s,
    )


def _build_code_describer(
    config: IndexerConfig,
    prompts: PromptRegistry | None,
    chat_model: Any | None,
    cache: ChunkCache | None = None,
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
        cache=cache,
        cache_enabled=config.description_cache_enabled,
        cache_ttl_s=config.description_cache_ttl_s,
    )
