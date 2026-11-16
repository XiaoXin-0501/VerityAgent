"""默认检索流水线。

按 RAG 文档 §2.2 / §9 的顺序执行：

```
权限 KB 过滤 → query 分析 → BM25 + vector 召回 → 图扩展（可选）
→ RRF 融合 → rerank → 文档状态/文档权限过滤 → 阈值/去重
→ 邻居扩展 → token 预算 → 文档描述/引用
```

设计边界：

- 权限、参数错误直接抛 ``RagError``；rerank/图/邻居等可降级阶段只记
  ``RetrievalWarning``，不静默失败；
- 文档状态过滤在 PG 中批量完成，避免重切片/删除期间旧 chunk 混入结果；
- 最终 `documents` 按 unique doc_id 批量补文档描述（D-103），不把描述塞进
  每个 chunk；
- 检索结果缓存 key 带权限摘要、索引版本、模型与过滤条件，避免跨权限/
  跨索引版本命中。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable, Iterable
from typing import Any
from uuid import UUID, uuid4

from config.rag import RetrieverConfig
from exception import RagError, RagErrorCode
from rag.indexer.enrichment import DomainProfile
from rag.retriever.cache import (
    RetrievalCache,
    document_meta_cache_key,
    document_status_cache_key,
    filters_digest,
    permission_digest,
    retrieval_cache_key,
)
from rag.retriever.fusion import rrf_fuse
from rag.retriever.graph import GraphExpander
from rag.retriever.hybrid import HybridRecallResult, HybridRecaller
from rag.retriever.metadata import RetrievalMetadataStore
from rag.retriever.neighbor import NeighborExpander
from rag.retriever.query import classify_intent_rule, estimate_language, normalize_query
from rag.retriever.rerank import Reranker
from rag.shared.enums import (
    RAGMode,
    RetrievalStage,
    RetrievalWarningCode,
)
from rag.shared.models import (
    Citation,
    DocumentMeta,
    GraphExpansionPolicy,
    HybridPolicy,
    NeighborPolicy,
    QueryAnalysis,
    RetrievalFilter,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStageTrace,
    RetrievalTrace,
    RetrievalWarning,
    RetrievedChunk,
    RerankPolicy,
    RagContext,
)
from utils.serialization import canonical_json
from utils.tokens import estimate_tokens
from utils.hashing import short_digest
from utils.time import utcnow

__all__ = ["DefaultRetrievalPipeline"]

logger = logging.getLogger(__name__)

# 文档状态过滤只接受 completed；与 PG DocumentStatus.COMPLETED 对齐。
_COMPLETED_STATUS = "completed"
# 图候选在 RRF 中的权重低于主召回，避免下一跳实体把原始 query 结果挤掉。
_GRAPH_RRF_WEIGHT = 0.5
# 引用片段最大字符数；只影响引用展示，不裁剪 chunk 正文。
_CITATION_SNIPPET_CHARS = 240

_CJK_TERM = re.compile(r"[一-鿿]{2,}")
_LATIN_TERM = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}")


def _warning(
    code: RetrievalWarningCode,
    message: str,
    *,
    stage: RetrievalStage | None = None,
    user_visible: bool = False,
    details: dict[str, Any] | None = None,
) -> RetrievalWarning:
    """构造结构化检索告警。"""
    return RetrievalWarning(
        code=code,
        message=message,
        stage=stage,
        user_visible=user_visible,
        details=details or {},
    )


def _stage(
    stages: list[RetrievalStageTrace],
    stage: RetrievalStage,
    started: float,
    *,
    input_count: int = 0,
    output_count: int = 0,
    metadata: dict[str, Any] | None = None,
) -> int:
    """记录一个阶段并返回耗时（毫秒）。"""
    elapsed = int((time.perf_counter() - started) * 1000)
    stages.append(
        RetrievalStageTrace(
            stage=stage,
            input_count=input_count,
            output_count=output_count,
            duration_ms=elapsed,
            metadata=metadata or {},
        )
    )
    return elapsed


def _chunk_tokens(chunk: RetrievedChunk) -> int:
    """取 chunk token 数；缺失时按正文估算，保证预算不会被绕过。"""
    value = chunk.metadata.get("token_count")
    if isinstance(value, int) and value >= 0:
        return value
    return estimate_tokens(chunk.content)


def _dedupe_ranking(chunks: Iterable[RetrievedChunk]) -> list[RetrievedChunk]:
    """按 chunk_id 去重并保留通道最高分；用于跨图 next-hop 合并。"""
    merged: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        current = merged.get(chunk.chunk_id)
        if current is None:
            merged[chunk.chunk_id] = chunk
            continue
        bm25 = max(
            (value for value in (current.bm25_score, chunk.bm25_score) if value is not None),
            default=None,
        )
        vector = max(
            (
                value
                for value in (current.vector_score, chunk.vector_score)
                if value is not None
            ),
            default=None,
        )
        merged[chunk.chunk_id] = current.model_copy(
            update={
                "bm25_score": bm25,
                "vector_score": vector,
                "metadata": {**current.metadata, **chunk.metadata},
            }
        )
    return sorted(
        merged.values(),
        key=lambda item: max(item.bm25_score or 0.0, item.vector_score or 0.0),
        reverse=True,
    )


def _merge_channel_scores(
    rankings: list[list[RetrievedChunk]],
) -> dict[str, dict[str, float | None]]:
    """汇总每个 chunk 在各通道的最高分，用于 RRF 后还原可观察分数。"""
    scores: dict[str, dict[str, float | None]] = {}
    for ranking in rankings:
        for chunk in ranking:
            entry = scores.setdefault(
                chunk.chunk_id, {"bm25": None, "vector": None}
            )
            if chunk.bm25_score is not None:
                entry["bm25"] = max(entry["bm25"] or 0.0, chunk.bm25_score)
            if chunk.vector_score is not None:
                entry["vector"] = max(entry["vector"] or 0.0, chunk.vector_score)
    return scores


def _snippet(content: str) -> str:
    """生成引用片段；压缩空白并限制长度，不改变 chunk 正文。"""
    value = re.sub(r"\s+", " ", content or "").strip()
    if len(value) <= _CITATION_SNIPPET_CHARS:
        return value
    return f"{value[:_CITATION_SNIPPET_CHARS].rstrip()}…"


def _dedupe_warnings(
    warnings: list[RetrievalWarning],
) -> list[RetrievalWarning]:
    """按 code/message/stage 去重，保留首次出现顺序。

    图扩展会对每个 next-hop 再跑一次召回，同一个 BM25/向量降级告警会重复
    出现；对外只保留一条，避免用户误以为发生了多次独立故障。
    """
    result: list[RetrievalWarning] = []
    seen: set[tuple[Any, str, Any]] = set()
    for item in warnings:
        key = (item.code, item.message, item.stage)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


class DefaultRetrievalPipeline:
    """RAG 默认检索流水线；依赖全部通过构造函数注入，便于单测。"""

    def __init__(
        self,
        config: RetrieverConfig | None = None,
        *,
        recaller: HybridRecaller,
        reranker: Reranker | None = None,
        graph_expander: GraphExpander | None = None,
        neighbor_expander: NeighborExpander | None = None,
        metadata_store: RetrievalMetadataStore | None = None,
        cache: RetrievalCache | None = None,
        domain_profile: DomainProfile | None = None,
        domain_profile_resolver: Callable[[str], DomainProfile] | None = None,
        query_rewriter: Any | None = None,
        intent_classifier: Any | None = None,
        max_query_chars: int = 2000,
    ) -> None:
        """保存检索配置与各阶段依赖；不主动建立外部连接。"""
        self.config = config or RetrieverConfig()
        self.recaller = recaller
        self.reranker = reranker
        self.graph_expander = graph_expander
        self.neighbor_expander = neighbor_expander
        self.metadata_store = metadata_store
        self.cache = cache
        self.domain_profile = domain_profile
        self.domain_profile_resolver = domain_profile_resolver
        self._domain_profile_cache: dict[str, DomainProfile] = (
            {domain_profile.name: domain_profile}
            if domain_profile is not None
            else {}
        )
        self.query_rewriter = query_rewriter
        self.intent_classifier = intent_classifier
        self.max_query_chars = max(1, int(max_query_chars))
        self.embedding_model_id = str(
            getattr(recaller, "embedding_model_id", "embedding")
        )
        self.embedding_version = str(
            getattr(recaller, "embedding_version", "v1")
        )
        self.rerank_model_id = (
            str(getattr(reranker, "model_id")) if reranker is not None else None
        )

    def _effective_top_k(self, request: RetrievalRequest) -> int:
        """请求 top_k 优先；未传时用 RetrieverConfig.top_k。"""
        return max(1, int(request.top_k or self.config.top_k))

    async def close(self) -> None:
        """释放流水线自身资源；connection manager 由应用生命周期统一关闭。"""
        return None

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """执行一次完整检索。"""
        started = time.perf_counter()
        trace_id = request.trace_id or uuid4().hex
        stages: list[RetrievalStageTrace] = []
        warnings: list[RetrievalWarning] = []

        query = request.query
        if len(query) > self.max_query_chars:
            warnings.append(
                _warning(
                    RetrievalWarningCode.QUERY_TOO_LONG,
                    f"query 超过 {self.max_query_chars} 字符，已截断",
                    stage=RetrievalStage.ANALYZE,
                    user_visible=True,
                    details={"original_chars": len(query)},
                )
            )
            query = query[: self.max_query_chars]

        # 1) KB 权限必须在任何召回前完成；全部无权限直接拒绝。
        permission_started = time.perf_counter()
        authorized_kbs = await self._authorize(request, warnings)
        _stage(
            stages,
            RetrievalStage.PERMISSION,
            permission_started,
            input_count=len(request.kb_ids),
            output_count=len(authorized_kbs),
        )

        # 2) query 规范化 + 语言/关键词/图 seed 实体。
        analyze_started = time.perf_counter()
        normalized = normalize_query(query)
        if not normalized:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "query 规范化后为空",
            )
        graph_policy = request.graph or self._default_graph_policy()
        if not graph_policy.allowed_relations:
            # 请求里只写 enabled=true 时沿用 RetrieverConfig 的强关系白名单；
            # 空列表不能静默变成“所有关系都允许”或“一个都不允许”。
            graph_policy = graph_policy.model_copy(
                update={
                    "allowed_relations": list(
                        self.config.graph_allowed_relations
                    )
                }
            )
        # 图扩展关闭时不需要读取 KB 画像，避免每次检索多查一次 PG。
        domain_profiles = (
            await self._load_domain_profiles(authorized_kbs, warnings)
            if graph_policy.enabled
            else ([self.domain_profile] if self.domain_profile else [])
        )
        analysis = self._analyze(request, normalized, domain_profiles)
        _stage(
            stages,
            RetrievalStage.ANALYZE,
            analyze_started,
            input_count=1,
            output_count=len(analysis.entities),
            metadata={
                "language": analysis.language,
                "entities": analysis.entities,
                "keywords": analysis.keywords,
            },
        )

        hybrid = request.hybrid or HybridPolicy(alpha=self.config.hybrid_alpha)
        rerank_policy = request.rerank or self._default_rerank_policy()
        neighbor_policy = request.neighbors or self._default_neighbor_policy()

        # 3) 索引版本 + 结果缓存；缓存命中直接返回，不触发任何外部召回。
        index_versions = await self._load_kb_versions(authorized_kbs, warnings)
        cache_key = self._build_cache_key(
            request=request,
            normalized=normalized,
            authorized_kbs=authorized_kbs,
            hybrid=hybrid,
            rerank_policy=rerank_policy,
            graph_policy=graph_policy,
            neighbor_policy=neighbor_policy,
            index_versions=index_versions,
            domain_profiles=domain_profiles,
        )
        cached = await self._read_cache(cache_key, warnings)
        if cached is not None:
            elapsed = int((time.perf_counter() - started) * 1000)
            merged_warnings = _dedupe_warnings(
                [*warnings, *cached.warnings]
            )
            trace = cached.trace.model_copy(
                update={
                    "trace_id": trace_id,
                    "cache_hit": True,
                    "created_at": utcnow(),
                    "warnings": [item.code.value for item in merged_warnings],
                }
            )
            return cached.model_copy(
                update={
                    "trace": trace,
                    "context": cached.context.model_copy(
                        update={"duration_ms": elapsed}
                    ),
                    "warnings": merged_warnings,
                    "duration_ms": elapsed,
                }
            )

        # 4) 缓存 miss 后才做 LLM 意图/改写，避免命中缓存时仍消耗 token。
        refine_started = time.perf_counter()
        analysis = await self._refine_analysis(analysis, warnings)
        if analysis.intent is not None or analysis.variants:
            _stage(
                stages,
                RetrievalStage.ANALYZE,
                refine_started,
                input_count=1,
                output_count=len(analysis.variants) + 1,
                metadata={
                    "intent": (
                        analysis.intent.value
                        if analysis.intent is not None
                        else None
                    ),
                    "variants": analysis.variants,
                    "query_rewrite_enabled": self.config.query_rewrite_enabled,
                },
            )

        # 5) 主召回：原 query + 改写变体；单变体失败只告警，主 query 失败仍抛错。
        recall_started = time.perf_counter()
        recalled = await self._recall_variants(
            [normalized, *analysis.variants],
            request=request,
            authorized_kbs=authorized_kbs,
            hybrid=hybrid,
            warnings=warnings,
        )
        warnings.extend(recalled.issues)
        for stage_name, stage in (
            ("BM25", RetrievalStage.BM25),
            ("VECTOR", RetrievalStage.VECTOR),
        ):
            elapsed_ms = recalled.timings_ms.get(stage_name)
            if elapsed_ms is None:
                continue
            stages.append(
                RetrievalStageTrace(
                    stage=stage,
                    input_count=1,
                    output_count=(
                        len(recalled.bm25)
                        if stage == RetrievalStage.BM25
                        else len(recalled.vector)
                    ),
                    duration_ms=elapsed_ms,
                )
            )
        if "VECTOR" in recalled.timings_ms:
            stages.append(
                RetrievalStageTrace(
                    stage=RetrievalStage.QUERY_EMBED,
                    input_count=1,
                    output_count=1,
                    duration_ms=recalled.timings_ms.get("QUERY_EMBED", 0),
                    metadata={
                        "cache_hit": recalled.query_embedding_cache_hit,
                    },
                )
            )
        if not recalled.timings_ms:
            _stage(
                stages,
                RetrievalStage.BM25
                if request.mode.value == "bm25"
                else RetrievalStage.VECTOR,
                recall_started,
                input_count=1,
                output_count=len(recalled.bm25) + len(recalled.vector),
            )

        # 5) 图扩展：先拿 next-hop 实体，再用它做 BM25/vector 召回。
        graph_bm25, graph_vector = await self._expand_graph(
            analysis=analysis,
            request=request,
            authorized_kbs=authorized_kbs,
            hybrid=hybrid,
            policy=graph_policy,
            warnings=warnings,
            stages=stages,
        )

        # 6) 文档状态/文档级权限过滤；图候选和主候选都必须过。
        status_started = time.perf_counter()
        all_candidates = _dedupe_ranking(
            [
                *recalled.bm25,
                *recalled.vector,
                *graph_bm25,
                *graph_vector,
            ]
        )
        allowed_ids, status_metadata = await self._filter_documents(
            all_candidates,
            request=request,
            index_versions=index_versions,
            warnings=warnings,
        )
        bm25 = [item for item in recalled.bm25 if item.chunk_id in allowed_ids]
        vector = [item for item in recalled.vector if item.chunk_id in allowed_ids]
        graph_bm25 = [item for item in graph_bm25 if item.chunk_id in allowed_ids]
        graph_vector = [item for item in graph_vector if item.chunk_id in allowed_ids]
        _stage(
            stages,
            RetrievalStage.DOC_STATUS,
            status_started,
            input_count=len(all_candidates),
            output_count=len(allowed_ids),
            metadata=status_metadata,
        )

        base_ids = {item.chunk_id for item in [*bm25, *vector]}
        graph_only_ids = {
            item.chunk_id
            for item in [*graph_bm25, *graph_vector]
            if item.chunk_id not in base_ids
        }

        # 7) RRF：两路基础召回 + 两路图召回；alpha 只调 BM25/vector 权重。
        fusion_started = time.perf_counter()
        rankings = [bm25, vector, graph_bm25, graph_vector]
        weights = [
            1.0 - hybrid.alpha,
            hybrid.alpha,
            (1.0 - hybrid.alpha) * _GRAPH_RRF_WEIGHT,
            hybrid.alpha * _GRAPH_RRF_WEIGHT,
        ]
        if request.mode.value == "bm25":
            weights = [1.0, 0.0, _GRAPH_RRF_WEIGHT, 0.0]
        elif request.mode.value == "vector":
            weights = [0.0, 1.0, 0.0, _GRAPH_RRF_WEIGHT]
        fused = rrf_fuse(
            rankings,
            key=lambda item: item.chunk_id,
            k=hybrid.rrf_k,
            weights=weights,
        )
        channel_scores = _merge_channel_scores(rankings)
        fused_chunks: list[RetrievedChunk] = []
        for chunk, score in fused:
            scores = channel_scores.get(chunk.chunk_id, {})
            metadata = dict(chunk.metadata)
            if chunk.chunk_id in graph_only_ids:
                metadata["graph_recalled"] = True
            fused_chunks.append(
                chunk.model_copy(
                    update={
                        "score": score,
                        "bm25_score": scores.get("bm25"),
                        "vector_score": scores.get("vector"),
                        "metadata": metadata,
                    }
                )
            )
        _stage(
            stages,
            RetrievalStage.FUSION,
            fusion_started,
            input_count=len(all_candidates),
            output_count=len(fused_chunks),
            metadata={
                "alpha": hybrid.alpha,
                "rrf_k": hybrid.rrf_k,
                "weights": weights,
            },
        )

        # 8) rerank；失败/未配置时保留融合分继续，不中断检索。
        ranked, rerank_stage = await self._rerank(
            normalized,
            fused_chunks,
            policy=rerank_policy,
            graph_policy=graph_policy,
            warnings=warnings,
        )
        stages.append(rerank_stage)

        # 9) 阈值、质量分、去重、top_k。
        filter_started = time.perf_counter()
        rerank_active = bool(
            rerank_stage.metadata.get("enabled")
            and not rerank_stage.metadata.get("failed")
        )
        selected = self._filter_and_select(
            ranked,
            request=request,
            rerank_active=rerank_active,
            warnings=warnings,
        )
        _stage(
            stages,
            RetrievalStage.FILTER,
            filter_started,
            input_count=len(ranked),
            output_count=len(selected),
            metadata={
                "min_score": self._effective_min_score(request),
                "min_quality_score": self._effective_min_quality(request),
            },
        )

        # 10) 邻居扩展；邻居只补上下文，不参与排序、不生成 citation。
        selected = await self._expand_neighbors(
            selected,
            policy=neighbor_policy,
            warnings=warnings,
            stages=stages,
        )

        # 11) token 预算；主 chunk 优先，超预算的邻居/尾部直接截断。
        selected = self._apply_budget(
            selected,
            request=request,
            warnings=warnings,
            stages=stages,
        )

        # 12) 按 unique doc_id 批量补文档描述（D-103），并生成引用。
        documents = await self._load_documents(
            selected,
            index_versions=index_versions,
            warnings=warnings,
            stages=stages,
        )
        selected = [
            chunk.model_copy(
                update={
                    "doc_name": (
                        documents[str(chunk.doc_id)].doc_name
                        if str(chunk.doc_id) in documents
                        else chunk.doc_name
                    )
                }
            )
            for chunk in selected
        ]
        citations: list[Citation] = []
        selected_with_citations: list[RetrievedChunk] = []
        for chunk in selected:
            if chunk.metadata.get("neighbor_of"):
                # 邻居保留无 citation，避免引用指向“上下文补充”而非命中片段。
                selected_with_citations.append(chunk)
                continue
            citation = self._build_citation(chunk)
            citations.append(citation)
            selected_with_citations.append(
                chunk.model_copy(update={"citation": citation})
            )

        elapsed = int((time.perf_counter() - started) * 1000)
        if not selected_with_citations:
            warnings.append(
                _warning(
                    RetrievalWarningCode.LOW_RECALL,
                    "本次检索没有可用片段",
                    stage=RetrievalStage.BUDGET,
                    user_visible=True,
                )
            )
        warnings = _dedupe_warnings(warnings)
        stage_timings = {item.stage.value: item.duration_ms for item in stages}
        trace = RetrievalTrace(
            trace_id=trace_id,
            stages=stages,
            candidate_count=len(all_candidates),
            final_count=len(selected_with_citations),
            timings_ms=stage_timings,
            alpha=hybrid.alpha,
            thresholds={
                "min_score": self._effective_min_score(request),
                "min_quality_score": self._effective_min_quality(request) or 0.0,
                "rerank_min_score": (
                    rerank_policy.min_score if rerank_policy.min_score is not None else 0.0
                ),
            },
            index_version=max(index_versions.values(), default=0),
            cache_hit=False,
            warnings=[item.code.value for item in warnings],
        )
        context = RagContext(
            query=request.query,
            kb_ids=authorized_kbs,
            chunks=selected_with_citations,
            documents=documents,
            top_k=self._effective_top_k(request),
            duration_ms=elapsed,
            mode=RAGMode.INJECTED,
        )
        result = RetrievalResult(
            query=request.query,
            normalized_query=normalized,
            kb_ids=authorized_kbs,
            mode=request.mode,
            chunks=selected_with_citations,
            documents=documents,
            context=context,
            citations=citations,
            trace=trace,
            warnings=warnings,
            duration_ms=elapsed,
        )
        await self._write_cache(cache_key, result)
        return result

    # ——————————————————— 权限 / 分析 ———————————————————

    async def _authorize(
        self,
        request: RetrievalRequest,
        warnings: list[RetrievalWarning],
    ) -> list[UUID]:
        """按 KB 的 ``required_permission_code`` 过滤；全部无权限直接拒绝。

        KB 权限码由 PG 的 ``knowledge_bases.required_permission_code`` 定义，
        常见值为 ``kb:read:<kb_id>`` 但允许部署自定义；不能只按 id 拼码。
        """
        if not request.kb_ids:
            raise RagError(
                RagErrorCode.INVALID_REQUEST,
                "检索请求至少需要一个 kb_id",
            )
        kb_ids = list(dict.fromkeys(request.kb_ids))
        codes: dict[UUID, str] = {}
        has_code_store = False
        if self.metadata_store is not None:
            loader = getattr(
                self.metadata_store,
                "get_kb_permission_codes",
                None,
            )
            if loader is not None:
                has_code_store = True
                try:
                    codes = await loader(kb_ids)
                except Exception as exc:  # noqa: BLE001 - 鉴权源失败必须拒绝
                    raise RagError(
                        RagErrorCode.DEPENDENCY_UNAVAILABLE,
                        "读取知识库权限码失败，检索已中止",
                        details={"cause": str(exc)},
                        retryable=True,
                    ) from exc
        authorized: list[UUID] = []
        filtered: list[str] = []
        for kb_id in kb_ids:
            # 没有权限码查询能力（单测/离线注入）时退回约定码；生产 store
            # 返回缺失说明 KB 不存在/已删除，必须拒绝，不能再用 id 拼码兜底。
            code = (
                codes.get(kb_id)
                if has_code_store
                else f"kb:read:{kb_id}"
            )
            if code and self._has_permission(request.permissions, code):
                authorized.append(kb_id)
            else:
                filtered.append(str(kb_id))
        if filtered:
            warnings.append(
                _warning(
                    RetrievalWarningCode.KB_PERMISSION_FILTERED,
                    "部分知识库无读取权限，已过滤",
                    stage=RetrievalStage.PERMISSION,
                    user_visible=True,
                    details={"kb_ids": filtered},
                )
            )
        if not authorized:
            raise RagError(
                RagErrorCode.PERMISSION_DENIED,
                "无权访问任何目标知识库",
                details={"kb_ids": [str(item) for item in request.kb_ids]},
            )
        return authorized

    @staticmethod
    def _has_permission(permissions: frozenset[str], required_code: str) -> bool:
        """检查权限码；``kb:read:*`` 作为部署级通配。"""
        return required_code in permissions or "kb:read:*" in permissions

    @staticmethod
    def _has_doc_permission(permissions: frozenset[str], doc_id: UUID) -> bool:
        """文档读权限支持精确码与通配码。"""
        return (
            f"doc:read:{doc_id}" in permissions
            or "doc:read:*" in permissions
        )

    def _doc_permission_enabled(self, request: RetrievalRequest) -> bool:
        """只有调用方显式传入 doc:read:* 权限时才做文档级权限过滤。

        这样默认 KB 级授权不会因为“没给每个 doc 的权限码”而误伤；一旦调用方
        使用文档级 ACL（权限集中出现 doc:read:），则严格按 doc_id 过滤。
        """
        if self.config.permission_mode != "KB_AND_DOC":
            return False
        return any(item.startswith("doc:read:") for item in request.permissions)

    def _analyze(
        self,
        request: RetrievalRequest,
        normalized: str,
        domain_profiles: list[DomainProfile],
    ) -> QueryAnalysis:
        """规范化 query 后的轻量分析：语言、关键词、图 seed 实体。"""
        entities: list[str] = []
        for profile in domain_profiles:
            for match in profile.match(normalized):
                if match.name not in entities:
                    entities.append(match.name)
        extra_entities = request.metadata.get("seed_entities") or request.metadata.get(
            "entities"
        )
        if isinstance(extra_entities, (list, tuple, set)):
            for item in extra_entities:
                value = str(item).strip()
                if value and value not in entities:
                    entities.append(value)
        keywords: list[str] = []
        for match in (*_CJK_TERM.findall(normalized), *_LATIN_TERM.findall(normalized)):
            value = match.strip()
            if value and value.lower() not in {item.lower() for item in keywords}:
                keywords.append(value)
        return QueryAnalysis(
            normalized_query=normalized,
            language=estimate_language(normalized),
            keywords=keywords[:20],
            entities=entities[:20],
            intent=(
                classify_intent_rule(normalized)
                if self.config.intent_rule_enabled
                else None
            ),
        )

    async def _refine_analysis(
        self,
        analysis: QueryAnalysis,
        warnings: list[RetrievalWarning],
    ) -> QueryAnalysis:
        """缓存 miss 后执行 LLM 意图兜底与 query 改写；失败只降级不阻断。"""
        intent = analysis.intent
        if (
            intent is None
            and self.config.intent_llm_enabled
            and self.intent_classifier is not None
        ):
            try:
                intent = await self.intent_classifier.classify(
                    analysis.normalized_query
                )
            except Exception as exc:  # noqa: BLE001 - 分类兜底可降级
                warnings.append(
                    _warning(
                        RetrievalWarningCode.INTENT_LLM_FAILED,
                        "意图 LLM 兜底失败，继续无意图检索",
                        stage=RetrievalStage.ANALYZE,
                        details={"cause": str(exc)},
                    )
                )
        elif self.config.intent_llm_enabled and self.intent_classifier is None:
            logger.debug("intent_llm_enabled=true 但未注入 intent classifier")

        variants = list(analysis.variants)
        if (
            self.config.query_rewrite_enabled
            and self.query_rewriter is not None
            and estimate_tokens(analysis.normalized_query)
            >= self.config.query_rewrite_min_query_tokens
        ):
            try:
                variants = await self.query_rewriter.rewrite(
                    analysis.normalized_query
                )
            except Exception as exc:  # noqa: BLE001 - 改写可降级
                warnings.append(
                    _warning(
                        RetrievalWarningCode.QUERY_REWRITE_FAILED,
                        "query 改写失败，使用原 query 检索",
                        stage=RetrievalStage.ANALYZE,
                        details={"cause": str(exc)},
                    )
                )
        elif self.config.query_rewrite_enabled and self.query_rewriter is None:
            warnings.append(
                _warning(
                    RetrievalWarningCode.QUERY_REWRITE_FAILED,
                    "query_rewrite_enabled=true 但未注入改写模型",
                    stage=RetrievalStage.ANALYZE,
                )
            )

        return analysis.model_copy(
            update={
                "intent": intent,
                "variants": variants[: self.config.query_rewrite_max_variants],
            }
        )

    # ——————————————————— 召回 / 图 ———————————————————

    async def _recall_variants(
        self,
        queries: list[str],
        *,
        request: RetrievalRequest,
        authorized_kbs: list[UUID],
        hybrid: HybridPolicy,
        warnings: list[RetrievalWarning],
    ) -> HybridRecallResult:
        """并发召回原 query 与改写变体；主 query 失败抛错，变体失败降级。"""
        # 主 query 单独调用，保持原有错误语义：主召回不可用应整体失败。
        primary = await self._recall(
            queries[0],
            request=request,
            authorized_kbs=authorized_kbs,
            hybrid=hybrid,
        )
        if len(queries) == 1:
            return primary

        results = await asyncio.gather(
            *[
                self._recall(
                    query,
                    request=request,
                    authorized_kbs=authorized_kbs,
                    hybrid=hybrid,
                )
                for query in queries[1:]
            ],
            return_exceptions=True,
        )
        bm25 = list(primary.bm25)
        vector = list(primary.vector)
        issues = list(primary.issues)
        timings = dict(primary.timings_ms)
        for query, result in zip(queries[1:], results, strict=True):
            if isinstance(result, BaseException):
                warnings.append(
                    _warning(
                        RetrievalWarningCode.QUERY_REWRITE_FAILED,
                        "改写查询召回失败，已跳过该变体",
                        stage=RetrievalStage.BM25,
                        details={"query": query, "cause": str(result)},
                    )
                )
                continue
            bm25.extend(result.bm25)
            vector.extend(result.vector)
            issues.extend(result.issues)
            for key, value in result.timings_ms.items():
                timings[key] = timings.get(key, 0) + value
        return HybridRecallResult(
            bm25=_dedupe_ranking(bm25),
            vector=_dedupe_ranking(vector),
            issues=issues,
            timings_ms=timings,
            query_embedding_cache_hit=primary.query_embedding_cache_hit,
        )

    async def _recall(
        self,
        query: str,
        *,
        request: RetrievalRequest,
        authorized_kbs: list[UUID],
        hybrid: HybridPolicy,
    ):
        """调用混合召回；底层异常统一映射为 RagError。"""
        try:
            return await self.recaller.recall(
                query,
                kb_ids=authorized_kbs,
                permissions=request.permissions,
                top_k_bm25=hybrid.top_k_bm25,
                top_k_vector=hybrid.top_k_vector,
                filters=request.filters,
                mode=request.mode.value,
            )
        except RagError:
            raise
        except Exception as exc:  # noqa: BLE001 - 召回失败必须对外收敛
            raise RagError(
                RagErrorCode.DEPENDENCY_UNAVAILABLE,
                "检索召回失败",
                details={"cause": str(exc)},
                retryable=True,
            ) from exc

    async def _expand_graph(
        self,
        *,
        analysis: QueryAnalysis,
        request: RetrievalRequest,
        authorized_kbs: list[UUID],
        hybrid: HybridPolicy,
        policy: GraphExpansionPolicy,
        warnings: list[RetrievalWarning],
        stages: list[RetrievalStageTrace],
    ) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
        """执行可选图扩展；返回图召回的 BM25/向量候选。"""
        if not policy.enabled:
            return [], []
        started = time.perf_counter()
        if self.graph_expander is None:
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_SKIPPED,
                    "图扩展已开启但未注入 GraphExpander",
                    stage=RetrievalStage.GRAPH,
                )
            )
            _stage(
                stages,
                RetrievalStage.GRAPH,
                started,
                input_count=len(analysis.entities),
                output_count=0,
                metadata={"skipped": True},
            )
            return [], []
        if not analysis.entities:
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_SKIPPED,
                    "query 未匹配到图 seed 实体",
                    stage=RetrievalStage.GRAPH,
                )
            )
            _stage(
                stages,
                RetrievalStage.GRAPH,
                started,
                input_count=0,
                output_count=0,
                metadata={"skipped": True},
            )
            return [], []

        try:
            next_hops = await self.graph_expander.expand(
                seed_entities=analysis.entities,
                kb_ids=[str(item) for item in authorized_kbs],
                policy=policy,
            )
        except Exception as exc:  # noqa: BLE001 - 图扩展可降级
            next_hops = []
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_FAILED,
                    "图扩展失败，已跳过下一跳召回",
                    stage=RetrievalStage.GRAPH,
                    details={"cause": str(exc)},
                )
            )
        last_error = getattr(self.graph_expander, "last_error", None)
        if last_error:
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_FAILED,
                    "图扩展失败，已跳过下一跳召回",
                    stage=RetrievalStage.GRAPH,
                    details={"cause": str(last_error)},
                )
            )
        elif not next_hops:
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_LOW_EVIDENCE,
                    "seed 实体没有满足证据阈值的下一跳",
                    stage=RetrievalStage.GRAPH,
                )
            )

        graph_bm25: list[RetrievedChunk] = []
        graph_vector: list[RetrievedChunk] = []
        failures = 0
        for next_hop in next_hops:
            try:
                recalled = await self.recaller.recall(
                    next_hop,
                    kb_ids=authorized_kbs,
                    permissions=request.permissions,
                    top_k_bm25=hybrid.top_k_bm25,
                    top_k_vector=hybrid.top_k_vector,
                    filters=request.filters,
                    mode=request.mode.value,
                )
            except Exception as exc:  # noqa: BLE001 - 单 next-hop 失败可降级
                failures += 1
                logger.warning("图 next-hop 召回失败 %s: %s", next_hop, exc)
                continue
            warnings.extend(recalled.issues)
            graph_bm25.extend(recalled.bm25)
            graph_vector.extend(recalled.vector)
        if next_hops and failures == len(next_hops):
            warnings.append(
                _warning(
                    RetrievalWarningCode.GRAPH_FAILED,
                    "所有图下一跳召回均失败",
                    stage=RetrievalStage.GRAPH,
                )
            )
        graph_bm25 = _dedupe_ranking(graph_bm25)
        graph_vector = _dedupe_ranking(graph_vector)
        _stage(
            stages,
            RetrievalStage.GRAPH,
            started,
            input_count=len(analysis.entities),
            output_count=len(next_hops),
            metadata={
                "next_hops": next_hops,
                "candidates": len(graph_bm25) + len(graph_vector),
            },
        )
        return graph_bm25, graph_vector

    # ——————————————————— 文档过滤 ———————————————————

    async def _filter_documents(
        self,
        candidates: list[RetrievedChunk],
        *,
        request: RetrievalRequest,
        index_versions: dict[UUID, int],
        warnings: list[RetrievalWarning],
    ) -> tuple[set[str], dict[str, Any]]:
        """批量过滤文档状态/文档级权限；返回允许的 chunk id 与统计。"""
        if not candidates:
            return set(), {"status_filtered": 0, "permission_filtered": 0}
        doc_ids = {chunk.doc_id for chunk in candidates}
        doc_kb = {chunk.doc_id: chunk.kb_id for chunk in candidates}
        allowed_docs = set(doc_ids)
        status_filtered = 0
        permission_filtered = 0

        if self.metadata_store is not None:
            statuses = await self._load_document_statuses(
                doc_kb,
                index_versions=index_versions,
                warnings=warnings,
            )
            allowed_docs = {
                doc_id
                for doc_id in doc_ids
                if str(statuses.get(doc_id, "")).lower() == _COMPLETED_STATUS
            }
            status_filtered = len(doc_ids) - len(allowed_docs)

        if self._doc_permission_enabled(request):
            before = set(allowed_docs)
            allowed_docs = {
                doc_id
                for doc_id in allowed_docs
                if self._has_doc_permission(request.permissions, doc_id)
            }
            permission_filtered = len(before) - len(allowed_docs)

        if status_filtered:
            warnings.append(
                _warning(
                    RetrievalWarningCode.DOC_STATUS_FILTERED,
                    "部分候选文档不是 completed 状态，已过滤",
                    stage=RetrievalStage.DOC_STATUS,
                    user_visible=True,
                    details={"filtered_docs": status_filtered},
                )
            )
        if permission_filtered:
            warnings.append(
                _warning(
                    RetrievalWarningCode.NO_PERMISSION,
                    "部分候选文档无读取权限，已过滤",
                    stage=RetrievalStage.DOC_STATUS,
                    user_visible=True,
                    details={"filtered_docs": permission_filtered},
                )
            )

        allowed_ids = {
            chunk.chunk_id
            for chunk in candidates
            if chunk.doc_id in allowed_docs
            and self._matches_post_filters(chunk, request.filters)
        }
        return allowed_ids, {
            "status_filtered": status_filtered,
            "permission_filtered": permission_filtered,
        }

    async def _load_document_statuses(
        self,
        doc_kb: dict[UUID, UUID],
        *,
        index_versions: dict[UUID, int],
        warnings: list[RetrievalWarning],
    ) -> dict[UUID, str]:
        """按 doc_id 读文档状态；Redis 可用时按 doc_id + 索引版本短缓存。"""
        if self.metadata_store is None or not doc_kb:
            return {}
        cached: dict[UUID, str] = {}
        misses: list[UUID] = list(doc_kb)
        cache_enabled = bool(
            self.cache is not None
            and self.config.cache_enabled
            and self.config.doc_status_cache_ttl_s > 0
        )
        cache = self.cache
        if cache_enabled:
            assert cache is not None
            misses = []
            for doc_id, kb_id in doc_kb.items():
                key = document_status_cache_key(
                    doc_id=str(doc_id),
                    index_version=index_versions.get(kb_id, 0),
                )
                try:
                    payload = await cache.get(key)
                except Exception as exc:  # noqa: BLE001 - 状态缓存失败回源 PG
                    warnings.append(
                        _warning(
                            RetrievalWarningCode.CACHE_ERROR,
                            "读取文档状态缓存失败，回源 PG",
                            stage=RetrievalStage.DOC_STATUS,
                            details={"cause": str(exc)},
                        )
                    )
                    payload = None
                status = (
                    payload.get("status")
                    if isinstance(payload, dict)
                    else payload if isinstance(payload, str) else None
                )
                if status:
                    cached[doc_id] = str(status)
                else:
                    misses.append(doc_id)

        if misses:
            try:
                from_store = await self.metadata_store.get_document_statuses(
                    misses
                )
            except Exception as exc:  # noqa: BLE001 - 状态过滤是安全边界，失败即拒绝
                raise RagError(
                    RagErrorCode.DEPENDENCY_UNAVAILABLE,
                    "读取文档状态失败，检索已中止",
                    details={"cause": str(exc)},
                    retryable=True,
                ) from exc
            cached.update(from_store)
            if cache_enabled:
                assert cache is not None
                for doc_id, status in from_store.items():
                    kb_id = doc_kb.get(doc_id)
                    if kb_id is None:
                        continue
                    key = document_status_cache_key(
                        doc_id=str(doc_id),
                        index_version=index_versions.get(kb_id, 0),
                    )
                    try:
                        await cache.set(
                            key,
                            {"status": status},
                            ttl_s=self.config.doc_status_cache_ttl_s,
                        )
                    except Exception:  # noqa: BLE001 - 写缓存失败不影响检索
                        pass
        return cached

    @staticmethod
    def _matches_post_filters(
        chunk: RetrievedChunk,
        filters: RetrievalFilter | None,
    ) -> bool:
        """对需要二次确认的元数据过滤做安全兜底。

        Weaviate 已下推大部分过滤；这里只处理可能由适配器版本差异导致的
        边界（page/created_after）以及图召回结果。tags 不在此判断，避免
        “schema 没有字段却静默按空值过滤”。
        """
        if filters is None:
            return True
        if filters.page_min is not None and (
            chunk.page_no is None or chunk.page_no < filters.page_min
        ):
            return False
        if filters.page_max is not None and (
            chunk.page_no is None or chunk.page_no > filters.page_max
        ):
            return False
        if filters.doc_ids and chunk.doc_id not in filters.doc_ids:
            return False
        return True

    # ——————————————————— rerank / 过滤 ———————————————————

    def _default_rerank_policy(self) -> RerankPolicy:
        """从 ``RetrieverConfig`` 构造默认 rerank 策略。"""
        return RerankPolicy(
            enabled=self.config.rerank_enabled,
            model_id=self.config.rerank_model_id,
            top_n=self.config.rerank_top_n,
            min_score=self.config.rerank_min_score,
            batch_size=self.config.rerank_batch_size,
        )

    def _default_graph_policy(self) -> GraphExpansionPolicy:
        """从 ``RetrieverConfig`` 构造默认图扩展策略。"""
        return GraphExpansionPolicy(
            enabled=self.config.graph_expansion_enabled,
            max_hops=self.config.graph_max_hops,
            allowed_relations=list(self.config.graph_allowed_relations),
            min_evidence=self.config.graph_min_evidence,
            max_next_hops=self.config.graph_max_next_hops,
            hub_as_intermediate=self.config.graph_hub_as_intermediate,
        )

    def _default_neighbor_policy(self) -> NeighborPolicy:
        """从 ``RetrieverConfig`` 构造默认邻居策略。"""
        return NeighborPolicy(
            enabled=self.config.neighbors_enabled,
            before=self.config.neighbor_before,
            after=self.config.neighbor_after,
            max_tokens=self.config.neighbor_max_tokens,
            same_doc=self.config.neighbor_same_doc,
            same_block_type=self.config.neighbor_same_block_type,
        )

    async def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        policy: RerankPolicy,
        graph_policy: GraphExpansionPolicy,
        warnings: list[RetrievalWarning],
    ) -> tuple[list[RetrievedChunk], RetrievalStageTrace]:
        """执行 rerank；未启用/未注入/失败都降级为融合排序。"""
        started = time.perf_counter()
        if not policy.enabled or not chunks:
            return chunks, RetrievalStageTrace(
                stage=RetrievalStage.RERANK,
                input_count=len(chunks),
                output_count=len(chunks),
                duration_ms=int((time.perf_counter() - started) * 1000),
                metadata={"enabled": False},
            )
        if self.reranker is None:
            warnings.append(
                _warning(
                    RetrievalWarningCode.RERANK_DISABLED,
                    "rerank 已开启但未注入 Reranker，使用融合分",
                    stage=RetrievalStage.RERANK,
                )
            )
            return chunks, RetrievalStageTrace(
                stage=RetrievalStage.RERANK,
                input_count=len(chunks),
                output_count=len(chunks),
                duration_ms=int((time.perf_counter() - started) * 1000),
                metadata={"enabled": False, "reason": "no_reranker"},
            )
        if (
            policy.model_id
            and self.rerank_model_id
            and policy.model_id != self.rerank_model_id
        ):
            warnings.append(
                _warning(
                    RetrievalWarningCode.RERANK_DISABLED,
                    "请求指定的 rerank 模型与已注入模型不一致，使用融合分",
                    stage=RetrievalStage.RERANK,
                    details={
                        "requested": policy.model_id,
                        "injected": self.rerank_model_id,
                    },
                )
            )
            return chunks, RetrievalStageTrace(
                stage=RetrievalStage.RERANK,
                input_count=len(chunks),
                output_count=len(chunks),
                duration_ms=int((time.perf_counter() - started) * 1000),
                metadata={"enabled": False, "reason": "model_mismatch"},
            )

        candidates = chunks[: max(1, min(policy.top_n, len(chunks)))]
        try:
            reranked = await self.reranker.rerank(query, candidates, policy)
        except Exception as exc:  # noqa: BLE001 - rerank 可降级
            warnings.append(
                _warning(
                    RetrievalWarningCode.RERANK_FAILED,
                    "rerank 失败，使用融合分",
                    stage=RetrievalStage.RERANK,
                    details={"cause": str(exc)},
                )
            )
            return chunks, RetrievalStageTrace(
                stage=RetrievalStage.RERANK,
                input_count=len(candidates),
                output_count=len(chunks),
                duration_ms=int((time.perf_counter() - started) * 1000),
                metadata={"enabled": True, "failed": True},
            )
        if not reranked:
            warnings.append(
                _warning(
                    RetrievalWarningCode.RERANK_FAILED,
                    "rerank 未返回任何候选，使用融合分",
                    stage=RetrievalStage.RERANK,
                )
            )
            return chunks, RetrievalStageTrace(
                stage=RetrievalStage.RERANK,
                input_count=len(candidates),
                output_count=len(chunks),
                duration_ms=int((time.perf_counter() - started) * 1000),
                metadata={"enabled": True, "failed": True, "empty": True},
            )

        # 图候选必须过图策略的 rerank 阈值；主召回只受全局 min_score 约束。
        filtered: list[RetrievedChunk] = []
        for chunk in reranked:
            updated = chunk.model_copy(update={"score": chunk.rerank_score})
            if (
                chunk.metadata.get("graph_recalled")
                and graph_policy.min_rerank_score is not None
                and (
                    chunk.rerank_score is None
                    or chunk.rerank_score < graph_policy.min_rerank_score
                )
            ):
                continue
            if (
                policy.min_score is not None
                and (
                    chunk.rerank_score is None
                    or chunk.rerank_score < policy.min_score
                )
            ):
                continue
            filtered.append(updated)
        if len(filtered) < len(candidates):
            warnings.append(
                _warning(
                    RetrievalWarningCode.QUALITY_FILTERED,
                    "部分候选未达到 rerank 阈值，已过滤",
                    stage=RetrievalStage.RERANK,
                    details={"filtered": len(candidates) - len(filtered)},
                )
            )
        return filtered, RetrievalStageTrace(
            stage=RetrievalStage.RERANK,
            input_count=len(candidates),
            output_count=len(filtered),
            duration_ms=int((time.perf_counter() - started) * 1000),
            metadata={
                "enabled": True,
                "model": self.rerank_model_id,
                "graph_min_rerank_score": graph_policy.min_rerank_score,
            },
        )

    def _effective_min_score(self, request: RetrievalRequest) -> float:
        """请求阈值优先，否则用配置默认值。"""
        if request.min_score is not None:
            return float(request.min_score)
        return float(self.config.min_score)

    def _effective_min_quality(self, request: RetrievalRequest) -> float | None:
        """请求质量阈值优先，否则用配置默认值。"""
        if request.min_quality_score is not None:
            return float(request.min_quality_score)
        if self.config.min_quality_score is None:
            return None
        return float(self.config.min_quality_score)

    def _filter_and_select(
        self,
        chunks: list[RetrievedChunk],
        *,
        request: RetrievalRequest,
        rerank_active: bool,
        warnings: list[RetrievalWarning],
    ) -> list[RetrievedChunk]:
        """阈值/质量分/内容去重后截取 top_k。

        排序分已在 rerank 阶段写入 ``score``；这里不再重复计算。只有 rerank
        真正成功时才用 ``rerank_top_k`` 封顶，失败降级时按请求 ``top_k`` 返回。
        """
        min_score = self._effective_min_score(request)
        min_quality = self._effective_min_quality(request)
        score_filtered = 0
        quality_filtered = 0
        deduped: list[RetrievedChunk] = []
        seen_hashes: set[str] = set()

        for chunk in chunks:
            if chunk.score is not None and chunk.score < min_score:
                score_filtered += 1
                continue
            if (
                min_quality is not None
                and chunk.quality_score is not None
                and chunk.quality_score < min_quality
            ):
                quality_filtered += 1
                continue
            # 内容 hash 优先；缺 hash 时用正文近似，避免同一内容不同 chunk id
            # 在图/近邻召回中重复进入上下文。
            content_hash = str(
                chunk.metadata.get("chunk_hash")
                or short_digest(chunk.content, 32)
            )
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            deduped.append(chunk)

        if score_filtered:
            warnings.append(
                _warning(
                    RetrievalWarningCode.QUALITY_FILTERED,
                    "部分候选低于最低分，已过滤",
                    stage=RetrievalStage.FILTER,
                    details={"filtered": score_filtered},
                )
            )
        if quality_filtered:
            warnings.append(
                _warning(
                    RetrievalWarningCode.QUALITY_FILTERED,
                    "部分候选低于最低质量分，已过滤",
                    stage=RetrievalStage.FILTER,
                    details={"filtered": quality_filtered},
                )
            )
        limit = self._effective_top_k(request)
        # rerank 成功时再受 RetrieverConfig.rerank_top_k 约束；未启用/失败降级
        # 时完全按调用方 top_k 返回，避免“配置里一个数、请求里一个数”暗冲突。
        if rerank_active:
            limit = min(limit, max(1, int(self.config.rerank_top_k)))
        return deduped[:limit]

    # ——————————————————— 邻居 / 预算 / 文档 ———————————————————

    async def _expand_neighbors(
        self,
        chunks: list[RetrievedChunk],
        *,
        policy: NeighborPolicy,
        warnings: list[RetrievalWarning],
        stages: list[RetrievalStageTrace],
    ) -> list[RetrievedChunk]:
        """追加邻居；邻居受策略 token 预算限制，不改变主 chunk 顺序。"""
        started = time.perf_counter()
        if not policy.enabled or not chunks:
            _stage(
                stages,
                RetrievalStage.NEIGHBOR,
                started,
                input_count=len(chunks),
                output_count=len(chunks),
                metadata={"enabled": policy.enabled},
            )
            return chunks
        if self.neighbor_expander is None:
            _stage(
                stages,
                RetrievalStage.NEIGHBOR,
                started,
                input_count=len(chunks),
                output_count=len(chunks),
                metadata={"enabled": False, "reason": "no_expander"},
            )
            return chunks
        try:
            neighbors = await self.neighbor_expander.expand(chunks, policy)
        except Exception as exc:  # noqa: BLE001 - 邻居可降级
            warnings.append(
                _warning(
                    RetrievalWarningCode.NEIGHBOR_FAILED,
                    "邻居扩展失败，仅使用命中片段",
                    stage=RetrievalStage.NEIGHBOR,
                    details={"cause": str(exc)},
                )
            )
            _stage(
                stages,
                RetrievalStage.NEIGHBOR,
                started,
                input_count=len(chunks),
                output_count=len(chunks),
                metadata={"failed": True},
            )
            return chunks

        used_tokens = 0
        kept: list[RetrievedChunk] = []
        main_ids = {chunk.chunk_id for chunk in chunks}
        seen_neighbors: set[str] = set()
        budget_exhausted = False
        for neighbor in neighbors:
            if neighbor.chunk_id in main_ids or neighbor.chunk_id in seen_neighbors:
                continue
            tokens = _chunk_tokens(neighbor)
            if used_tokens + tokens > policy.max_tokens:
                budget_exhausted = True
                warnings.append(
                    _warning(
                        RetrievalWarningCode.NEIGHBOR_BUDGET_EXCEEDED,
                        "邻居 token 预算已用完，部分邻居未加入上下文",
                        stage=RetrievalStage.NEIGHBOR,
                        details={
                            "max_tokens": policy.max_tokens,
                            "used_tokens": used_tokens,
                        },
                    )
                )
                break
            same_doc_mains = [
                item for item in chunks if item.doc_id == neighbor.doc_id
            ]
            if policy.same_doc and not same_doc_mains:
                continue
            if policy.same_block_type and not any(
                item.block_type == neighbor.block_type for item in same_doc_mains
            ):
                continue
            used_tokens += tokens
            seen_neighbors.add(neighbor.chunk_id)
            # 扩展器负责取邻居，但“邻居不能伪装成命中”由 pipeline 兜底：
            # 即使注入的扩展器忘了写 neighbor_of，也不会生成 citation。
            if not neighbor.metadata.get("neighbor_of"):
                nearest = min(
                    same_doc_mains,
                    key=lambda item: abs(
                        (item.chunk_index or 0) - (neighbor.chunk_index or 0)
                    ),
                    default=None,
                )
                neighbor = neighbor.model_copy(
                    update={
                        "score": None,
                        "citation": None,
                        "metadata": {
                            **neighbor.metadata,
                            **(
                                {"neighbor_of": nearest.chunk_id}
                                if nearest is not None
                                else {}
                            ),
                        },
                    }
                )
            kept.append(neighbor)
        if len(kept) < len(neighbors) and not budget_exhausted:
            warnings.append(
                _warning(
                    RetrievalWarningCode.NEIGHBOR_TRUNCATED,
                    "部分邻居因重复/边界条件未加入上下文",
                    stage=RetrievalStage.NEIGHBOR,
                    details={
                        "requested": len(neighbors),
                        "kept": len(kept),
                    },
                )
            )
        _stage(
            stages,
            RetrievalStage.NEIGHBOR,
            started,
            input_count=len(chunks),
            output_count=len(chunks) + len(kept),
            metadata={"neighbors": len(kept), "max_tokens": policy.max_tokens},
        )
        return [*chunks, *kept]

    def _apply_budget(
        self,
        chunks: list[RetrievedChunk],
        *,
        request: RetrievalRequest,
        warnings: list[RetrievalWarning],
        stages: list[RetrievalStageTrace],
    ) -> list[RetrievedChunk]:
        """按 token 预算截断；主 chunk 顺序保持，邻居排在主 chunk 之后。"""
        started = time.perf_counter()
        budget = (
            request.token_budget
            if request.token_budget is not None
            else (self.config.token_budget or self.config.max_context_tokens)
        )
        kept: list[RetrievedChunk] = []
        used = 0
        for chunk in chunks:
            tokens = _chunk_tokens(chunk)
            if kept and used + tokens > budget:
                continue
            kept.append(chunk)
            used += tokens
        if len(kept) < len(chunks):
            warnings.append(
                _warning(
                    RetrievalWarningCode.BUDGET_TRUNCATED,
                    "上下文 token 预算不足，部分片段已截断",
                    stage=RetrievalStage.BUDGET,
                    user_visible=True,
                    details={
                        "budget": budget,
                        "used": used,
                        "dropped": len(chunks) - len(kept),
                    },
                )
            )
        _stage(
            stages,
            RetrievalStage.BUDGET,
            started,
            input_count=len(chunks),
            output_count=len(kept),
            metadata={"budget": budget, "used_tokens": used},
        )
        return kept

    async def _load_documents(
        self,
        chunks: list[RetrievedChunk],
        *,
        index_versions: dict[UUID, int],
        warnings: list[RetrievalWarning],
        stages: list[RetrievalStageTrace],
    ) -> dict[str, DocumentMeta]:
        """按 unique doc_id 批量补文档描述（D-103），可选短 TTL 缓存。"""
        started = time.perf_counter()
        if not chunks:
            _stage(
                stages,
                RetrievalStage.DOC_META,
                started,
                input_count=0,
                output_count=0,
            )
            return {}
        if not self.config.include_doc_meta:
            _stage(
                stages,
                RetrievalStage.DOC_META,
                started,
                input_count=len(chunks),
                output_count=0,
                metadata={"skipped": True, "reason": "config_disabled"},
            )
            return {}
        if self.metadata_store is None:
            _stage(
                stages,
                RetrievalStage.DOC_META,
                started,
                input_count=len(chunks),
                output_count=0,
                metadata={"skipped": True},
            )
            return {}
        doc_kb = {chunk.doc_id: chunk.kb_id for chunk in chunks}
        doc_ids = list(doc_kb)
        doc_id_set = set(doc_ids)
        meta = await self._load_document_meta(
            doc_kb,
            index_versions=index_versions,
            warnings=warnings,
        )
        documents = {
            str(doc_id): item
            for doc_id, item in meta.items()
            if item.doc_id in doc_id_set
        }
        if len(documents) < len(doc_ids):
            warnings.append(
                _warning(
                    RetrievalWarningCode.DOC_META_MISSING,
                    "部分文档缺少元数据/描述",
                    stage=RetrievalStage.DOC_META,
                    details={
                        "expected": len(doc_ids),
                        "found": len(documents),
                    },
                )
            )
        _stage(
            stages,
            RetrievalStage.DOC_META,
            started,
            input_count=len(doc_ids),
            output_count=len(documents),
        )
        return documents

    async def _load_document_meta(
        self,
        doc_kb: dict[UUID, UUID],
        *,
        index_versions: dict[UUID, int],
        warnings: list[RetrievalWarning],
    ) -> dict[UUID, DocumentMeta]:
        """读文档描述；缓存 key 带 doc_id + KB 索引版本，描述更新后自然失效。"""
        if self.metadata_store is None:
            return {}
        cached: dict[UUID, DocumentMeta] = {}
        misses: list[UUID] = list(doc_kb)
        cache_enabled = bool(
            self.cache is not None
            and self.config.cache_enabled
            and self.config.doc_meta_cache_ttl_s > 0
        )
        cache = self.cache
        if cache_enabled:
            assert cache is not None
            misses = []
            for doc_id, kb_id in doc_kb.items():
                key = document_meta_cache_key(
                    doc_id=str(doc_id),
                    index_version=index_versions.get(kb_id, 0),
                )
                try:
                    payload = await cache.get(key)
                except Exception as exc:  # noqa: BLE001 - 描述缓存失败回源 PG
                    warnings.append(
                        _warning(
                            RetrievalWarningCode.CACHE_ERROR,
                            "读取文档描述缓存失败，回源 PG",
                            stage=RetrievalStage.DOC_META,
                            details={"cause": str(exc)},
                        )
                    )
                    payload = None
                if isinstance(payload, dict):
                    try:
                        cached[doc_id] = DocumentMeta.model_validate(payload)
                        continue
                    except Exception:  # noqa: BLE001 - 旧格式按 miss 处理
                        pass
                misses.append(doc_id)

        if misses:
            try:
                from_store = await self.metadata_store.get_document_meta(misses)
            except Exception as exc:  # noqa: BLE001 - 文档描述失败不影响召回正文
                warnings.append(
                    _warning(
                        RetrievalWarningCode.DOC_META_MISSING,
                        "读取文档描述失败，使用文件名/空描述",
                        stage=RetrievalStage.DOC_META,
                        details={"cause": str(exc)},
                    )
                )
                from_store = {}
            cached.update(from_store)
            if cache_enabled:
                assert cache is not None
                for doc_id, item in from_store.items():
                    kb_id = doc_kb.get(doc_id)
                    if kb_id is None:
                        continue
                    key = document_meta_cache_key(
                        doc_id=str(doc_id),
                        index_version=index_versions.get(kb_id, 0),
                    )
                    try:
                        await cache.set(
                            key,
                            item.model_dump(mode="json"),
                            ttl_s=self.config.doc_meta_cache_ttl_s,
                        )
                    except Exception:  # noqa: BLE001 - 写缓存失败不影响检索
                        pass
        return cached

    def _build_citation(self, chunk: RetrievedChunk) -> Citation:
        """由命中 chunk 构造引用；邻居不调用此方法。"""
        return Citation(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            kb_id=chunk.kb_id,
            doc_name=chunk.doc_name,
            page_no=chunk.page_no,
            section_type=(
                chunk.section_type.value if chunk.section_type is not None else None
            ),
            block_type=(
                chunk.block_type.value if chunk.block_type is not None else None
            ),
            chunk_index=chunk.chunk_index,
            snippet=_snippet(chunk.content),
            source_url=(
                str(chunk.metadata["source_url"])
                if chunk.metadata.get("source_url")
                else None
            ),
        )

    # ——————————————————— 缓存 ———————————————————

    def _resolve_domain_profile(self, name: str) -> DomainProfile | None:
        """按名称解析领域画像；失败回退全局画像并缓存。"""
        cached = self._domain_profile_cache.get(name)
        if cached is not None:
            return cached
        if self.domain_profile_resolver is None:
            return self.domain_profile
        try:
            profile = self.domain_profile_resolver(name)
        except Exception:  # noqa: BLE001 - 单 KB 画像失败不阻断检索
            logger.warning("加载知识库领域画像失败: %s", name, exc_info=True)
            return self.domain_profile
        self._domain_profile_cache[name] = profile
        return profile

    async def _load_domain_profiles(
        self,
        kb_ids: list[UUID],
        warnings: list[RetrievalWarning],
    ) -> list[DomainProfile]:
        """读取每个 KB 绑定的领域画像；NULL 回退全局默认。"""
        profiles: list[DomainProfile] = []
        loader = (
            getattr(self.metadata_store, "get_kb_domain_profiles", None)
            if self.metadata_store is not None
            else None
        )
        names: dict[UUID, str | None] = {}
        if callable(loader):
            try:
                # 元数据 store 的可选扩展；运行时按存在性探测。
                names = await loader(kb_ids)  # pyright: ignore[reportGeneralTypeIssues]
            except Exception as exc:  # noqa: BLE001 - 画像只影响可选图扩展
                warnings.append(
                    _warning(
                        RetrievalWarningCode.DOMAIN_PROFILE_FALLBACK,
                        "读取知识库领域画像失败，使用全局默认",
                        stage=RetrievalStage.ANALYZE,
                        details={"cause": str(exc)},
                    )
                )
        fallback = self.domain_profile
        for kb_id in kb_ids:
            name = names.get(kb_id) if names else None
            profile = self._resolve_domain_profile(name) if name else fallback
            if profile is not None and all(
                item.name != profile.name for item in profiles
            ):
                profiles.append(profile)
        return profiles

    async def _load_kb_versions(
        self,
        kb_ids: list[UUID],
        warnings: list[RetrievalWarning],
    ) -> dict[UUID, int]:
        """批量读取 KB 索引版本；失败时退化为 0，不阻断检索。"""
        if self.metadata_store is None:
            return {kb_id: 0 for kb_id in kb_ids}
        try:
            return await self.metadata_store.get_kb_index_versions(kb_ids)
        except Exception as exc:  # noqa: BLE001 - 版本只影响缓存 key
            warnings.append(
                _warning(
                    RetrievalWarningCode.INDEX_VERSION_MISMATCH,
                    "读取索引版本失败，缓存 key 使用 0",
                    stage=RetrievalStage.ANALYZE,
                    details={"cause": str(exc)},
                )
            )
            return {kb_id: 0 for kb_id in kb_ids}

    def _build_cache_key(
        self,
        *,
        request: RetrievalRequest,
        normalized: str,
        authorized_kbs: list[UUID],
        hybrid: HybridPolicy,
        rerank_policy: RerankPolicy,
        graph_policy: GraphExpansionPolicy,
        neighbor_policy: NeighborPolicy,
        index_versions: dict[UUID, int],
        domain_profiles: list[DomainProfile],
    ):
        """构造检索结果缓存 key；不缓存 query/正文明文。"""
        index_version = ",".join(
            f"{kb_id}:{index_versions.get(kb_id, 0)}"
            for kb_id in sorted(authorized_kbs, key=str)
        )
        # graph_version 在当前实现中没有独立版本号；把图策略/模式纳入 filters
        # 摘要，保证开启/关闭图扩展或换白名单时不会误命中旧结果。
        scope_digest = short_digest(
            canonical_json(
                {
                    "filters": filters_digest(request.filters),
                    "mode": request.mode.value,
                    "hybrid": hybrid.model_dump(mode="json"),
                    "graph": graph_policy.model_dump(mode="json"),
                    "neighbors": neighbor_policy.model_dump(mode="json"),
                    "rerank": rerank_policy.model_dump(mode="json"),
                    "domain_profiles": sorted(
                        profile.name for profile in domain_profiles
                    ),
                    "token_budget": (
                        request.token_budget
                        if request.token_budget is not None
                        else (self.config.token_budget or self.config.max_context_tokens)
                    ),
                    "include_doc_meta": self.config.include_doc_meta,
                    "rerank_top_k": self.config.rerank_top_k,
                    "rerank_enabled": bool(rerank_policy.enabled and self.reranker),
                    "min_score": self._effective_min_score(request),
                    "min_quality_score": self._effective_min_quality(request),
                    "query_rewrite": {
                        "enabled": self.config.query_rewrite_enabled,
                        "max_variants": self.config.query_rewrite_max_variants,
                    },
                    "intent": {
                        "rule": self.config.intent_rule_enabled,
                        "llm": self.config.intent_llm_enabled,
                    },
                }
            ),
            32,
        )
        return retrieval_cache_key(
            query=normalized,
            kb_ids=[str(item) for item in authorized_kbs],
            permission_digest=permission_digest(request.permissions),
            index_version=index_version or "0",
            embedding_model=f"{self.embedding_model_id}:{self.embedding_version}",
            rerank_model=(
                self.rerank_model_id
                if rerank_policy.enabled and self.reranker is not None
                else None
            ),
            top_k=self._effective_top_k(request),
            filters_digest=scope_digest,
        )

    async def _read_cache(
        self,
        key,
        warnings: list[RetrievalWarning],
    ) -> RetrievalResult | None:
        """读取检索缓存；payload 非法按 miss 处理。"""
        if not self.config.cache_enabled or self.config.cache_ttl_s <= 0:
            return None
        if self.cache is None:
            return None
        cache = self.cache
        try:
            payload = await cache.get(key)
        except Exception as exc:  # noqa: BLE001 - 缓存可降级
            warnings.append(
                _warning(
                    RetrievalWarningCode.CACHE_ERROR,
                    "读取检索缓存失败，回源检索",
                    stage=RetrievalStage.ANALYZE,
                    details={"cause": str(exc)},
                )
            )
            return None
        if payload is None:
            return None
        try:
            return RetrievalResult.model_validate(payload)
        except Exception:  # noqa: BLE001 - 旧格式/损坏值按 miss
            return None

    async def _write_cache(self, key, result: RetrievalResult) -> None:
        """写入检索缓存；失败不改变本次结果。"""
        if (
            not self.config.cache_enabled
            or self.config.cache_ttl_s <= 0
            or self.cache is None
        ):
            return
        cache = self.cache
        try:
            await cache.set(
                key,
                result.model_dump(mode="json"),
                ttl_s=self.config.cache_ttl_s,
            )
        except Exception as exc:  # noqa: BLE001 - 缓存写失败不影响返回
            logger.warning("写入检索缓存失败: %s", exc)
