"""Rerank 接口。"""

from __future__ import annotations

from typing import Protocol

from rag.shared.models import RerankPolicy, RetrievedChunk

__all__ = ["Reranker"]


class Reranker(Protocol):
    """重排协议；失败由检索流水线降级为 warning。"""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        policy: RerankPolicy,
        *,
        intent: str | None = None,
    ) -> list[RetrievedChunk]:
        """返回带 rerank_score 的 chunk；失败由 pipeline 降级为 warning。

        ``intent`` 是 Agent 查询规划产出的召回意图（LOOKUP/COMPARE/...），
        由实现方决定如何注入 provider 请求；不需要时不使用。
        """
        ...
