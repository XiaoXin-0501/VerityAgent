"""Weaviate 召回与邻居扩展实现。

该模块只负责“把结构化检索请求翻译成 Weaviate 查询”，不做权限判断、
融合、rerank 或预算裁剪；这些由 ``DefaultRetrievalPipeline`` 统一编排。

设计口径（RAG 文档 §9.5/§9.8）：

- BM25 按 query 主语言只查 ``content_zh`` 或 ``content_en`` 一个字段；
- 向量查询使用同一个 embedding 模型/维度，命中 query embedding 缓存；
- KB/元数据过滤必须下推 Weaviate；tags 当前 schema 没有对应属性，只记录
  告警，不静默假装已过滤；
- 邻居按 ``doc_id + chunk_index ±N`` 取，默认限制同 block_type，且不改变
  主 chunk 的 citation。
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any
from uuid import UUID

from exception import RagError, RagErrorCode
from rag.indexer.embedding import EmbeddingService
from rag.retriever.hybrid import HybridRecallResult
from rag.retriever.query import content_field_for_query, estimate_language
from rag.shared.enums import (
    BlockType,
    RetrievalStage,
    RetrievalWarningCode,
    SectionType,
)
from rag.shared.models import (
    NeighborPolicy,
    RetrievalFilter,
    RetrievalWarning,
    RetrievedChunk,
)
from store.weaviate import WeaviateStore

__all__ = ["WeaviateHybridRecaller", "WeaviateNeighborExpander"]

# DocChunk 查询需要回填的字段；与 schema 的检索契约保持一致。
DOC_CHUNK_PROPERTIES = [
    "kb_id",
    "doc_id",
    "doc_type",
    "section_type",
    "block_type",
    "heading",
    "heading_path",
    "language",
    "quality_score",
    "chunk_hash",
    "index_version",
    "chunking_version",
    "parser_id",
    "chunk_index",
    "page_no",
    "content_zh",
    "content_en",
    "semantic_description",
    "token_count",
    "created_at",
]


def _enum_value(value: Any) -> Any:
    """把 StrEnum / 普通值统一成落库字符串。"""
    return getattr(value, "value", value)


def _optional_enum(enum_type, value: Any):
    """把 Weaviate 字符串属性安全转回枚举；未知/空值返回 None。"""
    if value is None or value == "":
        return None
    try:
        return enum_type(_enum_value(value))
    except (TypeError, ValueError):
        return None


def _as_uuid(value: Any) -> UUID:
    """解析 Weaviate 中的 UUID 字符串；非法值直接报错，避免坏数据污染结果。"""
    return value if isinstance(value, UUID) else UUID(str(value))


def _chunk_from_object(obj: Any) -> RetrievedChunk:
    """把 Weaviate DataObject 投影为公共 ``RetrievedChunk``。

    ``score``/``bm25_score``/``vector_score`` 由调用方按通道填充；这里只
    还原正文、标题路径与可审计元数据。
    """
    props = dict(getattr(obj, "properties", None) or {})
    content = props.get("content_zh") or props.get("content_en") or ""
    heading_path = [
        str(item)
        for item in (props.get("heading_path") or [])
        if item is not None and str(item).strip()
    ]
    metadata = {
        "doc_type": props.get("doc_type") or None,
        "chunk_hash": props.get("chunk_hash") or None,
        "index_version": props.get("index_version"),
        "chunking_version": props.get("chunking_version") or None,
        "parser_id": props.get("parser_id") or None,
        "token_count": props.get("token_count"),
        "created_at": props.get("created_at"),
    }
    return RetrievedChunk(
        chunk_id=str(obj.uuid),
        doc_id=_as_uuid(props.get("doc_id")),
        kb_id=_as_uuid(props.get("kb_id")),
        chunk_index=(
            int(props["chunk_index"]) if props.get("chunk_index") is not None else None
        ),
        section_type=_optional_enum(SectionType, props.get("section_type")),
        block_type=_optional_enum(BlockType, props.get("block_type")),
        page_no=int(props["page_no"]) if props.get("page_no") is not None else None,
        content=str(content),
        heading=str(props["heading"]) if props.get("heading") else None,
        heading_path=heading_path,
        semantic_description=(
            str(props["semantic_description"])
            if props.get("semantic_description")
            else None
        ),
        language=str(props["language"]) if props.get("language") else None,
        quality_score=(
            float(props["quality_score"])
            if props.get("quality_score") is not None
            else None
        ),
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _build_where(
    kb_ids: list[Any],
    filters: RetrievalFilter | None,
) -> tuple[Any | None, list[str]]:
    """构造 Weaviate ``where``；返回 ``(where, 未实现过滤告警)``。

    所有能下推的过滤条件都在这里下推，避免“先召回再过滤”导致候选被
    无关 chunk 挤占。tags 在 DocChunk schema 中没有对应属性，只能显式告警。
    """
    from weaviate.classes.query import Filter

    clauses: list[Any] = []
    if kb_ids:
        clauses.append(
            Filter.by_property("kb_id").contains_any([str(item) for item in kb_ids])
        )
    warnings: list[str] = []
    if filters is not None:
        if filters.doc_ids:
            clauses.append(
                Filter.by_property("doc_id").contains_any(
                    [str(item) for item in filters.doc_ids]
                )
            )
        if filters.doc_types:
            clauses.append(
                Filter.by_property("doc_type").contains_any(
                    [_enum_value(item) for item in filters.doc_types]
                )
            )
        if filters.section_types:
            clauses.append(
                Filter.by_property("section_type").contains_any(
                    [_enum_value(item) for item in filters.section_types]
                )
            )
        if filters.block_types:
            clauses.append(
                Filter.by_property("block_type").contains_any(
                    [_enum_value(item) for item in filters.block_types]
                )
            )
        if filters.languages:
            clauses.append(
                Filter.by_property("language").contains_any(
                    [str(item) for item in filters.languages]
                )
            )
        if filters.page_min is not None:
            clauses.append(
                Filter.by_property("page_no").greater_or_equal(filters.page_min)
            )
        if filters.page_max is not None:
            clauses.append(
                Filter.by_property("page_no").less_or_equal(filters.page_max)
            )
        if filters.created_after is not None:
            clauses.append(
                Filter.by_property("created_at").greater_or_equal(
                    _date_value(filters.created_after)
                )
            )
        if filters.tags:
            warnings.append("tags 过滤未下推：DocChunk schema 暂未存储业务标签")

    if not clauses:
        return None, warnings
    where = clauses[0]
    for clause in clauses[1:]:
        where = where & clause
    return where, warnings


def _date_value(value: datetime) -> datetime:
    """Weaviate date 过滤直接使用 datetime；SDK 会按 ISO8601 序列化。"""
    return value


def _score_from_metadata(metadata: Any, *, vector: bool) -> float | None:
    """从 Weaviate Metadata 取通道分数。

    BM25 返回 ``score``；向量查询返回 ``certainty`` 或 ``distance``。距离
    越大越差，因此没有 certainty 时转成 ``1 - distance``，仅用于观察/融合前
    排序，不直接与 BM25 比较。
    """
    if metadata is None:
        return None
    if vector:
        certainty = getattr(metadata, "certainty", None)
        if certainty is not None:
            return float(certainty)
        distance = getattr(metadata, "distance", None)
        if distance is not None:
            return max(0.0, 1.0 - float(distance))
        return None
    score = getattr(metadata, "score", None)
    return float(score) if score is not None else None


class WeaviateHybridRecaller:
    """基于 Weaviate DocChunk 的 BM25 + 向量召回器。"""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        vector_store: WeaviateStore | None = None,
        dimensions: int | None = None,
        embedding_version: str = "v1",
    ) -> None:
        """保存 embedding 服务与 Weaviate store；collection 懒获取。"""
        self.embedding = embedding_service
        self.store = vector_store or WeaviateStore()
        self.dimensions = dimensions
        self.embedding_version = embedding_version

    @property
    def embedding_model_id(self) -> str:
        """缓存 key / trace 使用的 embedding 模型标识。"""
        return self.embedding.model_id

    async def recall(
        self,
        query: str,
        *,
        kb_ids: list[Any],
        permissions: frozenset[str],
        top_k_bm25: int,
        top_k_vector: int,
        filters: RetrievalFilter | None = None,
        mode: str = "hybrid",
    ) -> HybridRecallResult:
        """按模式召回 BM25/向量候选；单通道失败在 HYBRID 下降级并告警。"""
        where, filter_warnings = _build_where(kb_ids, filters)
        issues = [
            RetrievalWarning(
                code=RetrievalWarningCode.LOW_RECALL,
                message=message,
                stage=RetrievalStage.FILTER,
            )
            for message in filter_warnings
        ]
        timings: dict[str, int] = {}
        bm25: list[RetrievedChunk] = []
        vector: list[RetrievedChunk] = []
        embedding_cache_hit = False

        tasks: list[tuple[str, asyncio.Task]] = []
        if mode in {"hybrid", "bm25"} and top_k_bm25 > 0:
            tasks.append(
                (
                    "bm25",
                    asyncio.create_task(
                        self._bm25(query, where=where, limit=top_k_bm25)
                    ),
                )
            )
        if mode in {"hybrid", "vector"} and top_k_vector > 0:
            tasks.append(
                (
                    "vector",
                    asyncio.create_task(
                        self._vector(query, where=where, limit=top_k_vector)
                    ),
                )
            )
        results: list[Any] = []
        if tasks:
            results = await asyncio.gather(
                *(task for _name, task in tasks),
                return_exceptions=True,
            )

        failures: list[str] = []
        for (name, _task), result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                failures.append(f"{name}: {result}")
                code = (
                    RetrievalWarningCode.EMBEDDING_FAILED
                    if name == "vector"
                    else RetrievalWarningCode.LOW_RECALL
                )
                issues.append(
                    RetrievalWarning(
                        code=code,
                        message=(
                            "向量召回失败，已降级为 BM25 结果"
                            if name == "vector"
                            else "BM25 召回失败，已降级为向量结果"
                        ),
                        stage=(
                            RetrievalStage.VECTOR
                            if name == "vector"
                            else RetrievalStage.BM25
                        ),
                        details={"cause": str(result)},
                    )
                )
                continue
            if name == "bm25":
                chunks, elapsed, _cache_hit = result
                timings["BM25"] = elapsed
                bm25 = chunks
            else:
                chunks, elapsed, cache_hit, embed_elapsed = result
                timings["VECTOR"] = elapsed
                timings["QUERY_EMBED"] = embed_elapsed
                vector = chunks
                embedding_cache_hit = cache_hit

        if failures and len(failures) == len(tasks):
            # 所有请求的通道都失败：调用方无法得到可用候选，直接上抛为依赖错误。
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "Weaviate 召回失败",
                details={"failures": failures},
                retryable=True,
            )
        # 空结果不等同于异常：常见原因是中文分词器未启用/查询词不在词典。
        # 显式写 LOW_RECALL，避免检索“看起来成功但 BM25 实际没工作”。
        bm25_failed = any(item.startswith("bm25:") for item in failures)
        vector_failed = any(item.startswith("vector:") for item in failures)
        if (
            mode in {"hybrid", "bm25"}
            and not bm25
            and top_k_bm25 > 0
            and not bm25_failed
        ):
            message = "BM25 未返回候选"
            if estimate_language(query) == "zh":
                message += (
                    "；中文查询请确认 Weaviate 已启用 "
                    "ENABLE_TOKENIZER_GSE_CH=true，且 collection 使用 gse_ch"
                )
            issues.append(
                RetrievalWarning(
                    code=RetrievalWarningCode.LOW_RECALL,
                    message=message,
                    stage=RetrievalStage.BM25,
                )
            )
        if (
            mode in {"hybrid", "vector"}
            and not vector
            and top_k_vector > 0
            and not vector_failed
        ):
            issues.append(
                RetrievalWarning(
                    code=RetrievalWarningCode.LOW_RECALL,
                    message="向量召回未返回候选",
                    stage=RetrievalStage.VECTOR,
                )
            )
        return HybridRecallResult(
            bm25=bm25,
            vector=vector,
            issues=issues,
            timings_ms=timings,
            query_embedding_cache_hit=embedding_cache_hit,
        )

    async def _bm25(
        self,
        query: str,
        *,
        where: Any | None,
        limit: int,
    ) -> tuple[list[RetrievedChunk], int, bool]:
        """执行 BM25；返回候选、耗时（毫秒）与固定 False 的 embedding 命中。"""
        from weaviate.classes.query import MetadataQuery

        field = content_field_for_query(query)
        started = time.perf_counter()
        response = await self.store.maybe_to_thread(
            self.store.doc_chunks.query.bm25,
            query=query,
            query_properties=[field],
            limit=max(1, int(limit)),
            filters=where,
            return_properties=DOC_CHUNK_PROPERTIES,
            return_metadata=MetadataQuery(score=True),
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        chunks: list[RetrievedChunk] = []
        for obj in getattr(response, "objects", []) or []:
            chunk = _chunk_from_object(obj)
            score = _score_from_metadata(getattr(obj, "metadata", None), vector=False)
            chunks.append(
                chunk.model_copy(
                    update={
                        "bm25_score": score,
                        "metadata": {**chunk.metadata, "bm25_score": score},
                    }
                )
            )
        return chunks, elapsed, False

    async def _vector(
        self,
        query: str,
        *,
        where: Any | None,
        limit: int,
    ) -> tuple[list[RetrievedChunk], int, bool, int]:
        """执行 query embedding + near_vector；返回候选、耗时、缓存命中与向量耗时。"""
        from weaviate.classes.query import MetadataQuery

        started = time.perf_counter()
        embed_started = time.perf_counter()
        embedding = await self.embedding.embed_query(
            query,
            dimensions=self.dimensions,
        )
        embedding_ms = int((time.perf_counter() - embed_started) * 1000)
        if not embedding.vectors:
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "query embedding 为空",
                retryable=True,
            )
        response = await self.store.maybe_to_thread(
            self.store.doc_chunks.query.near_vector,
            near_vector=embedding.vectors[0],
            limit=max(1, int(limit)),
            filters=where,
            return_properties=DOC_CHUNK_PROPERTIES,
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        chunks: list[RetrievedChunk] = []
        for obj in getattr(response, "objects", []) or []:
            chunk = _chunk_from_object(obj)
            score = _score_from_metadata(getattr(obj, "metadata", None), vector=True)
            chunks.append(
                chunk.model_copy(
                    update={
                        "vector_score": score,
                        "metadata": {**chunk.metadata, "vector_score": score},
                    }
                )
            )
        return chunks, elapsed, embedding.cache_hits > 0, embedding_ms


class WeaviateNeighborExpander:
    """按 ``doc_id + chunk_index`` 取前后邻居；不跨文档、不改变主 chunk 引用。"""

    def __init__(
        self,
        *,
        vector_store: WeaviateStore | None = None,
    ) -> None:
        """保存 Weaviate store；collection 懒获取。"""
        self.store = vector_store or WeaviateStore()

    async def expand(
        self,
        chunks: list[RetrievedChunk],
        policy: NeighborPolicy,
    ) -> list[RetrievedChunk]:
        """返回去重后的邻居 chunk；主 chunk 自身不重复返回。

        邻居的 ``score`` 置空、``citation`` 置空，避免在引用中伪装成命中；
        ``metadata.neighbor_of`` 指向主 chunk，便于 Prompt 层理解上下文来源。
        """
        if not policy.enabled or (policy.before <= 0 and policy.after <= 0):
            return []
        result: list[RetrievedChunk] = []
        seen: set[str] = set()
        for chunk in chunks:
            if chunk.chunk_index is None:
                continue
            neighbors = await self._fetch_neighbors(chunk, policy)
            for neighbor in neighbors:
                if neighbor.chunk_id == chunk.chunk_id or neighbor.chunk_id in seen:
                    continue
                seen.add(neighbor.chunk_id)
                result.append(
                    neighbor.model_copy(
                        update={
                            "score": None,
                            "citation": None,
                            "metadata": {
                                **neighbor.metadata,
                                "neighbor_of": chunk.chunk_id,
                            },
                        }
                    )
                )
        return result

    async def _fetch_neighbors(
        self,
        chunk: RetrievedChunk,
        policy: NeighborPolicy,
    ) -> list[RetrievedChunk]:
        """查询单条主 chunk 的邻居；失败由 pipeline 降级为告警。"""
        from weaviate.classes.query import Filter

        index = int(chunk.chunk_index or 0)
        lower = max(0, index - max(0, policy.before))
        upper = index + max(0, policy.after)
        where = Filter.by_property("doc_id").equal(str(chunk.doc_id))
        where = where & Filter.by_property("chunk_index").greater_or_equal(lower)
        where = where & Filter.by_property("chunk_index").less_or_equal(upper)
        response = await self.store.maybe_to_thread(
            self.store.doc_chunks.query.fetch_objects,
            filters=where,
            limit=max(1, policy.before + policy.after + 1),
            return_properties=DOC_CHUNK_PROPERTIES,
        )
        neighbors: list[RetrievedChunk] = []
        for obj in getattr(response, "objects", []) or []:
            neighbor = _chunk_from_object(obj)
            if policy.same_doc and neighbor.doc_id != chunk.doc_id:
                continue
            if (
                policy.same_block_type
                and neighbor.block_type != chunk.block_type
            ):
                continue
            neighbors.append(neighbor)
        neighbors.sort(key=lambda item: item.chunk_index or 0)
        return neighbors
