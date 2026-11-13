"""RAG 依赖装配入口。

门面只接收 ``RagDependencies``；默认实现（Weaviate/Neo4j/MinIO/PG）在这里
集中接线，避免 facade 反向依赖具体存储。
"""

from __future__ import annotations

from typing import Any

from config.rag import IndexerConfig

from .facade import Indexer, RagDependencies, create_indexer
from .indexer import (
    DefaultIndexingPipeline,
    EmbeddingService,
    InMemoryJobStore,
    IndexingRepository,
    JobStore,
    MinioManifestStore,
    MinioSourceStore,
    Neo4jGraphIndexStore,
    PgIndexingRepository,
    RuleGraphExtractor,
    TextIngestionPipeline,
    WeaviateChunkStore,
)

__all__ = [
    "build_dependencies",
    "build_default_indexing_pipeline",
    "build_default_indexer",
    "ensure_default_vector_schema",
]


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
    graph_extractor: Any | None = None,
    description_refiner: Any | None = None,
    worker_id: str = "rag-indexer",
    run_in_background: bool = True,
) -> DefaultIndexingPipeline:
    """装配默认 v1 入库流水线。

    只构造对象，不建立连接；首次写 Weaviate/Neo4j/MinIO/PG 时才真正连接。
    ``vector_store.ensure_schema()`` 应在应用启动时单独调用（dev/test 可
    自动建 collection，prod 只校验）。
    """
    cfg = config or IndexerConfig()
    return DefaultIndexingPipeline(
        cfg,
        ingestion=TextIngestionPipeline(
            cfg,
            description_refiner=description_refiner,
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
        graph_extractor=graph_extractor or RuleGraphExtractor(cfg),
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
