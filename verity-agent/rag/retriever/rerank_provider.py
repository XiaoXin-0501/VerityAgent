"""Provider rerank 适配器。

协议层只定义 ``Reranker``；本模块把 ``provider.reranker.RerankModel`` 适配
成检索流水线可用的实现：

- 输入只使用 ``query + chunk.content``，不使用入库时的 ``embedding_text``，
  避免表格/代码的派生文本与正文产生偏差；
- 按 provider 能力与请求 ``batch_size`` 分批，失败直接上抛，由 pipeline
  降级为 ``RERANK_FAILED`` 并保留融合排序；
- provider 返回的 id 与候选 chunk_id 对齐；漏返回的候选保留在原顺序末尾，
  不静默丢弃。
"""

from __future__ import annotations

from typing import Any

from provider.reranker.types import RerankDocument, RerankRequest
from rag.shared.models import RerankPolicy, RetrievedChunk

__all__ = ["ProviderReranker"]


class ProviderReranker:
    """RerankModel 的业务适配器。"""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str | None = None,
    ) -> None:
        """保存模型句柄；model_id 用于缓存 key/trace，不改变 provider 调用。"""
        if not hasattr(model, "rerank"):
            raise TypeError("rerank model 必须提供 rerank()")
        self.model = model
        spec = getattr(model, "spec", None)
        self.model_id = model_id or getattr(spec, "id", None) or "rerank"
        capabilities = getattr(spec, "rerank_capabilities", None)
        self.max_batch_size = int(
            getattr(capabilities, "max_batch_size", 32) or 32
        )

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        policy: RerankPolicy,
        *,
        intent: str | None = None,
    ) -> list[RetrievedChunk]:
        """返回带 ``rerank_score`` 的候选，按分数降序；未返回项排在末尾。

        ``intent`` 由 Agent 查询规划提供；provider 只接受 query+documents，
        因此把意图作为 query 前缀一起送入，让模型知道“这次是按事实查询、
        比较还是步骤方法排序”。
        """
        if not chunks:
            return []
        rerank_query = _with_intent(query, intent)
        batch_size = max(1, min(int(policy.batch_size), self.max_batch_size))
        scores: dict[str, float] = {}
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            response = await self.model.rerank(
                RerankRequest(
                    model=self.model_id,
                    query=rerank_query,
                    documents=[
                        RerankDocument(id=chunk.chunk_id, text=chunk.content)
                        for chunk in batch
                    ],
                    # 输出数量由 pipeline 的 rerank_top_n 控制；这里让 provider
                    # 返回本批全部候选，便于统一处理漏返回项。
                    top_k=None,
                    return_documents=False,
                )
            )
            for result in response.results:
                scores[str(result.id)] = float(result.score)

        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            score = scores.get(chunk.chunk_id)
            ranked.append(
                chunk.model_copy(
                    update={
                        "rerank_score": score,
                        "metadata": {
                            **chunk.metadata,
                            **({"rerank_score": score} if score is not None else {}),
                        },
                    }
                )
            )
        ranked.sort(
            key=lambda item: (
                item.rerank_score is not None,
                item.rerank_score or 0.0,
            ),
            reverse=True,
        )
        return ranked


# 意图 → 中文提示；未知值直接原样使用，保留上游扩展空间。
_INTENT_HINTS = {
    "LOOKUP": "查询事实",
    "COMPARE": "比较对象",
    "SUMMARIZE": "总结内容",
    "HOWTO": "步骤方法",
}


def _with_intent(query: str, intent: str | None) -> str:
    """把召回意图作为轻量前缀注入 rerank query。"""
    value = (intent or "").strip()
    if not value:
        return query
    hint = _INTENT_HINTS.get(value.upper(), value)
    return f"[查询意图：{hint}] {query}"
