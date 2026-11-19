"""查询规划器：规则短路 + LLM 规划。

调用顺序：

1. 规则判断高置信“不需要 RAG”的场景（寒暄/闲聊/个人记忆/纯计算）；
2. 其余请求在开启 LLM 时调用 ``LLMQueryPlanner``，输入用户画像、
   会话历史、垂直领域目录/实体类型参考和规则结论；规则判定的需要 RAG
   只作为先验，不跳过 LLM，以便补全指代、提取实体、扩写用户意图；
3. LLM 失败/超时/输出非法时保留规则兜底结果，不阻断对话。

规划结果由 Agent 主循环消费：改写 query 与 RAG 意图传给 RAG，
用户意图用于长期事实召回。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from uuid import UUID

from config.agent import QueryPlanConfig

from .llm import LLMQueryPlanner, QueryPlanResult
from .models import QueryPlan, QueryPlanSource
from .rules import build_rule_plan, evaluate_rules, normalize_query

__all__ = ["QueryPlanner"]

logger = logging.getLogger(__name__)


class QueryPlanner:
    """规则短路“明确不需要 RAG”，其余交给 LLM 的查询规划器。"""

    def __init__(
        self,
        config: QueryPlanConfig | None = None,
        *,
        llm: object | None = None,
        llm_model_id: str | None = None,
        domain_terms: Iterable[str] = (),
        domain_entity_types: str | None = None,
        domain_noise_terms: str | None = None,
        prompts: object | None = None,
    ) -> None:
        """保存配置、可选 LLM、领域词与实体类型/噪声词参考；不发起网络请求。"""
        self.config = config or QueryPlanConfig()
        self.domain_terms = tuple(
            item for item in (str(term).strip() for term in domain_terms) if item
        )
        self.llm_planner: LLMQueryPlanner | None = None
        model_id = self.config.model_id or llm_model_id
        if self.config.enabled and self.config.use_llm and llm and model_id:
            self.llm_planner = LLMQueryPlanner(
                llm,
                model_id=model_id,
                timeout_s=self.config.timeout_s,
                max_output_tokens=self.config.max_output_tokens,
                entity_types=_clip(
                    domain_entity_types,
                    self.config.domain_entity_types_chars,
                ),
                noise_terms=_clip(
                    domain_noise_terms,
                    self.config.domain_noise_terms_chars,
                ),
                prompts=prompts,  # type: ignore[arg-type]
            )

    @property
    def enabled(self) -> bool:
        """是否启用规划。"""
        return bool(self.config.enabled)

    async def plan(
        self,
        original_query: str,
        *,
        profile_text: str | None = None,
        session_summary: str | None = None,
        history_text: str | None = None,
        domain_catalog: str | None = None,
        trace_id: str,
        conversation_id: UUID,
        turn_id: str,
        user_id: UUID,
    ) -> QueryPlanResult:
        """返回规划结果；任何异常都降级为规则结果。"""
        normalized = normalize_query(original_query)
        if not self.config.enabled:
            return QueryPlanResult(
                QueryPlan(
                    original_query=original_query,
                    normalized_query=normalized,
                    needs_rag=False,
                    rewritten_query=None,
                    rag_intent=None,
                    user_intent=normalized,
                    source=QueryPlanSource.DISABLED,
                    confidence=1.0,
                    reason="查询规划未启用",
                )
            )
        rule = evaluate_rules(normalized, domain_terms=self.domain_terms)
        # 规则只对“明确不需要 RAG”做终态短路（寒暄/个人记忆/纯计算）。
        # 明确需要 RAG 的规则命中仍交给 LLM：只有 LLM 能补全指代、提取
        # 图 seed 实体，并结合画像/历史扩写 user_intent。LLM 不可用时再
        # 回退到规则结果，保持原有降级能力。
        if (
            (rule.decided and not rule.needs_rag)
            or not self.config.use_llm
            or self.llm_planner is None
            or len(normalized) < self.config.min_query_chars
        ):
            return QueryPlanResult(
                build_rule_plan(
                    original_query,
                    normalized_query=normalized,
                    rule=rule,
                    source=QueryPlanSource.RULE,
                )
            )

        prompt_domain = _clip(domain_catalog, self.config.domain_catalog_chars)
        try:
            result = await self.llm_planner.plan(
                original_query,
                normalized_query=normalized,
                profile_text=_clip_text(profile_text, self.config.profile_token_budget),
                session_summary=session_summary,
                history_text=history_text,
                domain_catalog=prompt_domain,
                rule=rule,
                trace_id=trace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                user_id=user_id,
            )
        except Exception:  # noqa: BLE001 - 规划失败不阻断对话
            logger.warning("query 规划异常，回退规则结果", exc_info=True)
            result = None
        if result is not None:
            return result
        # LLM 失败/超时：规则未决策时由 build_rule_plan 做低置信兜底。
        return QueryPlanResult(
            build_rule_plan(
                original_query,
                normalized_query=normalized,
                rule=rule,
                source=QueryPlanSource.FALLBACK,
                confidence=0.4,
            )
        )


def _clip(value: str | None, limit: int) -> str | None:
    """按字符数截断目录；limit<=0 表示不注入。"""
    if not value or limit <= 0:
        return None
    text = value.strip()
    return text[:limit] if len(text) > limit else text


def _clip_text(value: str | None, token_limit: int) -> str | None:
    """按 token 估算截断画像；token_limit<=0 表示不注入。"""
    if not value or token_limit <= 0:
        return None
    from utils.tokens import truncate_to_tokens

    return truncate_to_tokens(value.strip(), token_limit)
