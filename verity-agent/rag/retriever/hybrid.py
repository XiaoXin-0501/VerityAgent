"""混合召回接口（BM25 + vector）。

v1 由 retrieval pipeline 注入具体 Weaviate/Embedding 实现；本模块只定义
统一的候选结果形状，避免 fusion/rerank 直接依赖 store。
"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RetrievalWarning, RetrievedChunk

__all__ = ["HybridRecallResult", "HybridRecaller"]


class HybridRecallResult:
    """一次混合召回的候选：BM25、向量、告警与阶段耗时分开保留。"""

    def __init__(
        self,
        *,
        bm25: list[RetrievedChunk] | None = None,
        vector: list[RetrievedChunk] | None = None,
        warnings: list[str] | None = None,
        issues: list[RetrievalWarning] | None = None,
        timings_ms: dict[str, int] | None = None,
        query_embedding_cache_hit: bool = False,
    ) -> None:
        """保存两路候选、告警与阶段耗时；空值统一成空列表/空字典。

        ``warnings`` 保留为纯字符串（兼容早期实现/测试）；``issues`` 是带
        code/stage 的结构化告警，pipeline 优先使用它。
        """
        self.bm25 = bm25 or []
        self.vector = vector or []
        self.warnings = warnings or []
        self.issues = issues or []
        self.timings_ms = timings_ms or {}
        self.query_embedding_cache_hit = query_embedding_cache_hit


class HybridRecaller(Protocol):
    """混合召回协议；具体 Weaviate 实现由检索流水线注入。"""

    async def recall(
        self,
        query: str,
        *,
        kb_ids: list,
        permissions: frozenset[str],
        top_k_bm25: int,
        top_k_vector: int,
        filters=None,
        mode: str = "hybrid",
        explain: bool = False,
    ) -> HybridRecallResult:
        """返回 BM25 与向量两路候选；权限过滤必须在召回前完成。

        ``explain=True`` 时 BM25 额外请求 explainScore，用于规则路径下的
        seed 实体提取；默认关闭，避免不必要的响应体积。
        """
        ...
