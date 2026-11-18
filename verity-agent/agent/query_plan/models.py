"""Agent 查询规划的数据模型。

规划产物只描述“这次请求要做什么”，不包含答案：

- ``rewritten_query``：需要 RAG 时给检索使用的改写 query；
- ``rag_intent``：需要 RAG 时给 rerank 使用的召回意图；
- ``user_intent``：用户真正想完成的事，用于召回长期记忆事实；
- ``needs_rag``：是否进行知识库召回。

规则初筛与 LLM 兜底使用同一模型，``source`` 标明结果来源，便于日志与排障。
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

    @property
    def title(self) -> str:
        """给 rerank/日志使用的中文短标签。"""
        return {
            self.LOOKUP: "查询事实",
            self.COMPARE: "比较对象",
            self.SUMMARIZE: "总结内容",
            self.HOWTO: "步骤方法",
        }[self]


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
    # 需要 RAG 时才非空；不需要时为 None
    rag_intent: RagRecallIntent | None = None
    # 用户真正想完成的事；用于长期事实 BM25 召回
    user_intent: str = ""
    source: QueryPlanSource = QueryPlanSource.RULE
    # 规划置信度；规则命中/LLM 解析成功时为高值，兜底为低值
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    # 判断依据；只进日志
    reason: str = ""

    @property
    def retrieval_query(self) -> str:
        """召回消费的 query：改写优先，其次规范化原话。"""
        return (self.rewritten_query or "").strip() or self.normalized_query

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
