"""RAG 公共门面：只暴露 Indexer / Retriever 与公共 DTO。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from config.rag import IndexerConfig, RetrieverConfig
from exception import RagError, RagErrorCode

from .indexer.pipeline import IndexingPipeline
from .retriever.pipeline import RetrievalPipeline
from .shared.models import (
    IndexDocumentRequest,
    IndexingJobView,
    RetrievalRequest,
    RetrievalResult,
)

__all__ = [
    "RagDependencies",
    "Indexer",
    "Retriever",
    "create_indexer",
    "create_retriever",
]


@dataclass(slots=True)
class RagDependencies:
    """RAG 外部依赖装配。

    具体实现由 bootstrap 注入；门面不直接 import store/parser 实现。
    """

    # 入库流水线实现
    indexing_pipeline: IndexingPipeline | None = None
    # 检索流水线实现
    retrieval_pipeline: RetrievalPipeline | None = None
    # 可选的 embedding/rerank 模型句柄
    embedding_model: Any | None = None
    rerank_model: Any | None = None
    # 可选存储实现
    vector_store: Any | None = None
    graph_store: Any | None = None
    manifest_store: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Indexer:
    """知识库写入门面；实际流水线由 dependencies 注入。"""

    def __init__(self, config: IndexerConfig, pipeline: IndexingPipeline) -> None:
        self.config = config
        self._pipeline = pipeline

    async def index(self, request: IndexDocumentRequest) -> IndexingJobView:
        return await self._pipeline.submit(request)

    async def get_job(self, job_id: str) -> IndexingJobView | None:
        return await self._pipeline.get_job(job_id)

    async def cancel(self, job_id: str) -> None:
        await self._pipeline.cancel(job_id)

    async def delete(self, doc_id: UUID) -> IndexingJobView:
        return await self._pipeline.delete(doc_id)

    async def close(self) -> None:
        await self._pipeline.close()


class Retriever:
    """知识库召回门面；实际流水线由 dependencies 注入。"""

    def __init__(self, config: RetrieverConfig, pipeline: RetrievalPipeline) -> None:
        self.config = config
        self._pipeline = pipeline

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        return await self._pipeline.retrieve(request)

    async def close(self) -> None:
        await self._pipeline.close()


async def create_indexer(
    config: IndexerConfig,
    *,
    dependencies: RagDependencies | None = None,
) -> Indexer:
    deps = dependencies or RagDependencies()
    if deps.indexing_pipeline is None:
        raise RagError(
            RagErrorCode.DEPENDENCY_UNAVAILABLE,
            "IndexingPipeline 未注入，无法创建 Indexer",
        )
    return Indexer(config, deps.indexing_pipeline)


async def create_retriever(
    config: RetrieverConfig,
    *,
    dependencies: RagDependencies | None = None,
) -> Retriever:
    deps = dependencies or RagDependencies()
    if deps.retrieval_pipeline is None:
        raise RagError(
            RagErrorCode.DEPENDENCY_UNAVAILABLE,
            "RetrievalPipeline 未注入，无法创建 Retriever",
        )
    return Retriever(config, deps.retrieval_pipeline)
