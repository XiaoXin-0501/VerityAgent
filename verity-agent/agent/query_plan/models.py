"""Agent 查询规划的数据模型。

规划产物只描述“这次请求要做什么”，不包含答案：

- ``rewritten_query``：需要 RAG 时给检索使用的改写 query（主改写，
  与用户原话同语言，用于展示/兼容）；
- ``rewritten_query_zh`` / ``rewritten_query_en``：同一语义的中英文检索
  变体；RAG 会分别召回，避免中文问句只能命中中文文档、英文问句只能
  命中英文文档；
- ``rag_intent``：需要 RAG 时给 rerank 使用的召回意图；
- ``user_intent``：用户真正想完成的事，用于召回长期记忆事实；
- ``needs_rag``：是否进行知识库召回。

规则先验与 LLM 规划使用同一模型，``source`` 标明结果来源，便于日志与排障。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "QueryPlan",
    "QueryPlanSource",
    "RagRecallIntent",
    "parse_rag_intent",
]


class _Model(BaseModel):
    """不可变值对象；未知字段直接报错，避免 LLM 输出污染运行态。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RagRecallIntent(StrEnum):
    """RAG 召回意图；只用于检索/rerank，不进入最终回答。"""

    LOOKUP = "LOOKUP"
    COMPARE = "COMPARE"
    SUMMARIZE = "SUMMARIZE"
    HOWTO = "HOWTO"


class QueryPlanSource(StrEnum):
    """规划结果来源。"""

    DISABLED = "disabled"
    RULE = "rule"
    LLM = "llm"
    FALLBACK = "fallback"


class QueryPlan(_Model):
    """一次用户请求的查询规划结果。"""

    original_query: str
    # 规范化后的原话；规则/LLM 都基于它工作
    normalized_query: str
    needs_rag: bool = False
    # 需要 RAG 时才非空；不需要时为 None
    rewritten_query: str | None = None
    # 同一语义的中英文检索变体；LLM 规划时尽量都产出，规则路径可为空。
    # 分别召回后再融合，避免单一语言 query 漏掉另一语言文档。
    rewritten_query_zh: str | None = None
    rewritten_query_en: str | None = None
    # 需要 RAG 时才非空；不需要时为 None
    rag_intent: RagRecallIntent | None = None
    # 用户真正想完成的事；用于长期事实 BM25 召回
    user_intent: str = ""
    # 需要 RAG 时由图/实体链接使用的实体；规划器尽量中英文各一条，
    # 可结合 user_intent 补充少量直接相关实体，不要求穷举领域。
    entities: list[str] = Field(default_factory=list)
    source: QueryPlanSource = QueryPlanSource.RULE
    # 规划置信度；规则命中/LLM 解析成功时为高值，兜底为低值
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 判断依据；只进日志
    reason: str = ""

    @property
    def retrieval_query(self) -> str:
        """召回消费的主 query：改写优先，其次规范化原话。"""
        return (self.rewritten_query or "").strip() or self.normalized_query

    @property
    def retrieval_queries(self) -> list[str]:
        """返回本次检索要并发召回的全部 query。

        顺序固定为“主改写 → 中文变体 → 英文变体”，去空、去重；
        没有改写时退化为规范化原话，保证调用方总能拿到至少一条。
        """
        queries: list[str] = []
        for value in (
            self.rewritten_query,
            self.rewritten_query_zh,
            self.rewritten_query_en,
        ):
            text = (value or "").strip()
            if text and text not in queries:
                queries.append(text)
        return queries or [self.normalized_query]

    @property
    def fact_query(self) -> str:
        """长期事实召回 query：用户意图优先，其次规范化原话。"""
        return (self.user_intent or "").strip() or self.normalized_query

    @property
    def rag_intent_label(self) -> str | None:
        """给 RAG 层传递的意图标签；不需要 RAG 时为 None。"""
        return self.rag_intent.value if self.rag_intent is not None else None


def parse_rag_intent(value: Any) -> RagRecallIntent | None:
    """宽松解析 LLM 输出；只接受枚举内值，其他返回 None。"""
    text = str(value or "").strip().upper()
    if not text or text in {"NONE", "NULL", "NO", "UNKNOWN"}:
        return None
    for item in RagRecallIntent:
        if text == item.value or text == item.name:
            return item
    return None
