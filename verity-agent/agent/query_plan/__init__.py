"""Agent 查询规划：意图识别、query 改写与 RAG 召回决策。

规则初筛在前，LLM 兜底在后；产物统一为 ``QueryPlan``：

- ``rewritten_query``：交给 RAG 召回消费；
- ``rag_intent``：交给 RAG rerank 消费；
- ``user_intent``：交给长期事实召回消费。

RAG 模块不再执行 query 改写/意图识别，只消费本模块的结果。
"""

from .llm import LLMQueryPlanner, QueryPlanResult
from .models import QueryPlan, QueryPlanSource, RagRecallIntent, parse_rag_intent
from .planner import QueryPlanner
from .rules import (
    RuleResult,
    build_rule_plan,
    evaluate_rules,
    fallback_needs_rag,
    normalize_query,
)

__all__ = [
    "LLMQueryPlanner",
    "QueryPlan",
    "QueryPlanResult",
    "QueryPlanSource",
    "QueryPlanner",
    "RagRecallIntent",
    "RuleResult",
    "build_rule_plan",
    "evaluate_rules",
    "fallback_needs_rag",
    "normalize_query",
    "parse_rag_intent",
]
