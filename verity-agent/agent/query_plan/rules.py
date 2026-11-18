"""查询规划的规则初筛。

规则只做低成本、可解释的高置信判断：

- 寒暄/致谢/能力询问/纯计算/个人记忆 → 不需要 RAG；
- 明确的查资料/文档/知识库表达，或 compare/summarize/howto/lookup 意图，
  或命中垂直领域词 → 需要 RAG；
- 其余交给 LLM 兜底，避免规则误判。

规则无法做出改写，因此命中 RAG 时只把规范化后的原话作为 ``rewritten_query``；
真正的指代消解和改写交给 LLM。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from utils.text import normalize_whitespace

from .models import QueryPlan, QueryPlanSource, RagRecallIntent

__all__ = [
    "RuleResult",
    "build_rule_plan",
    "estimate_language",
    "fallback_needs_rag",
    "normalize_query",
    "evaluate_rules",
]

_CJK = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")

# 明确不需要知识库的场景：寒暄、闲聊、能力/身份、个人记忆、纯计算。
_NO_RAG_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*(你好|您好|hi|hello|hey|在吗|早上好|晚上好|晚安|再见|拜拜)\s*[!！。.]*$",
        r"^\s*(谢谢|多谢|thanks|thank you|好的|收到|明白了|ok)\s*[!！。.]*$",
        r"(你是谁|你叫什么|你能做什么|你会什么|你是什么模型|who are you|what can you do)",
        r"(你怎么看|你觉得|你认为|你的看法|你建议|你推荐|你更喜欢|你偏好)",
        r"(我叫什么|你记得我|还记得我|我的偏好|我之前说过|我们上次|我的名字是什么)",
        r"^\s*[\d\s+\-*/().=？?]+\s*$",
        r"^\s*(今天几号|现在几点|今天星期几|当前时间)\s*[?？。]*$",
    )
)

# 明确的“去查资料”表达；命中即认为需要 RAG。
_RAG_CUE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(知识库|文档|资料|文献|手册|指南|说明书|病例库|资料库)",
        r"(查一下|查查|查找|检索|搜索|搜一下|帮我找|找一下|翻一下)",
        r"(根据|依据|按照).{0,12}(文档|资料|指南|说明书|知识库)",
        r"(文档里|资料里|书里|指南里|知识库里).{0,20}(说|写|提到|规定)",
        r"(search|look up|find).{0,20}(document|knowledge|manual|guideline)",
    )
)

# 高置信检索意图；与原 RAG 规则口径一致，但结果改由 Agent 产出。
_INTENT_RULES: tuple[tuple[RagRecallIntent, tuple[str, ...]], ...] = (
    (
        RagRecallIntent.COMPARE,
        (
            "对比", "比较", "区别", "差异", "哪个好", "哪个更",
            "difference", "compare", "versus", " vs ", "vs.", "better",
        ),
    ),
    (
        RagRecallIntent.SUMMARIZE,
        (
            "总结", "概括", "摘要", "综述", "主要讲", "讲了什么",
            "summarize", "summary", "overview", "key points",
        ),
    ),
    (
        RagRecallIntent.HOWTO,
        (
            "如何", "怎么", "怎样", "步骤", "流程", "治疗方法", "预防",
            "how to", "how do", "steps", "procedure", "treatment",
        ),
    ),
    (
        RagRecallIntent.LOOKUP,
        (
            "是什么", "定义", "含义", "what is", "definition", "meaning",
        ),
    ),
)

_QUESTION_RE = re.compile(
    r"[?？]|^\s*(什么|怎么|如何|为什么|为啥|哪|是否|能否|多少|几|谁|何时|哪里)"
)


@dataclass(frozen=True, slots=True)
class RuleResult:
    """规则初筛结果；``needs_rag`` 为空表示规则无法高置信判断。"""

    needs_rag: bool | None
    rag_intent: RagRecallIntent | None
    user_intent: str
    reason: str

    @property
    def decided(self) -> bool:
        """规则是否已给出明确结论。"""
        return self.needs_rag is not None


def normalize_query(query: str) -> str:
    """全半角归一、空白压缩；保留原始大小写用于实体匹配。"""
    value = unicodedata.normalize("NFKC", query or "")
    return normalize_whitespace(value, keep_newlines=False)


def estimate_language(text: str) -> str | None:
    """粗粒度语言判断：zh / en；混合或无法判断返回 None。"""
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if cjk and cjk >= latin:
        return "zh"
    if latin and latin > cjk:
        return "en"
    return None


def evaluate_rules(
    query: str,
    *,
    domain_terms: Iterable[str] = (),
) -> RuleResult:
    """规则初筛；返回是否需要 RAG、召回意图与用户意图。

    ``domain_terms`` 是垂直领域画像中的补充词/标准名，用于识别领域内提问；
    规则只检查 query 是否包含这些词，不做实体抽取。
    """
    value = f" {query.strip()} "
    lowered = value.lower()

    for pattern in _NO_RAG_PATTERNS:
        if pattern.search(value):
            return RuleResult(
                needs_rag=False,
                rag_intent=None,
                user_intent=query,
                reason="命中无需知识库的寒暄/闲聊/个人记忆规则",
            )

    intent = _classify_intent(value)
    if any(pattern.search(value) for pattern in _RAG_CUE_PATTERNS):
        return RuleResult(
            needs_rag=True,
            rag_intent=intent or RagRecallIntent.LOOKUP,
            user_intent=query,
            reason="用户明确要求查资料/文档/知识库",
        )

    if intent is not None:
        return RuleResult(
            needs_rag=True,
            rag_intent=intent,
            user_intent=query,
            reason=f"命中高置信检索意图规则: {intent.value}",
        )

    for term in domain_terms:
        text = str(term).strip()
        if len(text) >= 2 and text.lower() in lowered:
            return RuleResult(
                needs_rag=True,
                rag_intent=RagRecallIntent.LOOKUP,
                user_intent=query,
                reason=f"命中垂直领域词: {text}",
            )

    return RuleResult(
        needs_rag=None,
        rag_intent=None,
        user_intent=query,
        reason="规则无法高置信判断，交给 LLM 兜底",
    )


def fallback_needs_rag(query: str) -> bool:
    """LLM 不可用时的保守兜底：像知识提问就检索，否则不检索。

    规则已经识别出的寒暄/个人记忆在前一步返回，不会走到这里；这里只处理
    “无法判断”的 query，用问句特征做最后一道低置信判断。
    """
    text = query.strip()
    if len(text) < 2:
        return False
    return bool(_QUESTION_RE.search(text))


def build_rule_plan(
    original_query: str,
    *,
    normalized_query: str | None = None,
    rule: RuleResult | None = None,
    source: QueryPlanSource = QueryPlanSource.RULE,
    confidence: float = 0.8,
) -> QueryPlan:
    """把规则结果投影成 ``QueryPlan``。

    规则已决策时直接采用；未决策时按 ``fallback_needs_rag`` 兜底，来源标记
    为 ``FALLBACK``。需要 RAG 时用规范化原话作为改写 query。
    """
    normalized = normalized_query or normalize_query(original_query)
    decision = rule or evaluate_rules(normalized)
    needs_rag = decision.needs_rag
    if needs_rag is None:
        needs_rag = fallback_needs_rag(normalized)
        source = QueryPlanSource.FALLBACK
        confidence = min(confidence, 0.5)
    return QueryPlan(
        original_query=original_query,
        normalized_query=normalized,
        needs_rag=bool(needs_rag),
        rewritten_query=normalized if needs_rag else None,
        rag_intent=(
            decision.rag_intent or RagRecallIntent.LOOKUP if needs_rag else None
        ),
        user_intent=(decision.user_intent or normalized).strip(),
        source=source,
        confidence=confidence if needs_rag or decision.decided else 0.4,
        reason=decision.reason,
    )


def _classify_intent(value: str) -> RagRecallIntent | None:
    """规则识别高置信 query 意图；无法判断返回 None。"""
    lowered = value.lower()
    for intent, needles in _INTENT_RULES:
        if any(needle in lowered for needle in needles):
            return intent
    return None
