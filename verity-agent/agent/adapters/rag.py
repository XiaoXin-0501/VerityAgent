"""RAG Retriever -> Agent RagPort 适配器。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from rag.shared.models import RetrievalRequest

__all__ = ["RetrieverRagAdapter"]


class RetrieverRagAdapter:
    """把 RAG 门面包装成 Agent INJECTED 模式所需的窄端口。"""

    def __init__(self, retriever: Any) -> None:
        """保存只读 Retriever；生命周期由应用管理。"""
        self.retriever = retriever

    async def retrieve(
        self,
        *,
        query: str,
        kb_ids: list[UUID],
        user_id: UUID,
        permissions: frozenset[str],
        top_k: int | None = None,
        trace_id: str | None = None,
        original_query: str | None = None,
        rewritten_query: str | None = None,
        rewritten_queries: list[str] | None = None,
        rag_intent: str | None = None,
        entities: list[str] | None = None,
    ) -> Any:
        """执行一次只读检索；权限过滤由 RAG 流水线负责。"""
        return await self.retriever.retrieve(
            RetrievalRequest(
                query=query,
                original_query=original_query,
                rewritten_query=rewritten_query,
                rewritten_queries=list(rewritten_queries or []),
                rag_intent=rag_intent,
                entities=list(entities or []),
                kb_ids=list(kb_ids),
                user_id=user_id,
                permissions=permissions,
                top_k=top_k,
                trace_id=trace_id,
            )
        )
