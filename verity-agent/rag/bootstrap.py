"""RAG 依赖装配入口。

具体 parser/chunker/store/embedding 实现在后续里程碑接入；这里只提供
可注入的依赖容器，避免门面绑定实现。
"""

from __future__ import annotations

from typing import Any

from .facade import RagDependencies

__all__ = ["build_dependencies"]


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
