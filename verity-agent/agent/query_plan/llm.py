"""查询规划的 LLM 兜底。

规则无法高置信判断时才走到这里。LLM 输入包含：

- 用户画像（长期偏好/背景）；
- 当前会话历史（最近若干条）；
- 垂直领域目录（主画像 + 知识库绑定子画像）；
- 用户原话与规则初筛结论。

LLM 输出短 JSON：是否需要 RAG、改写 query、RAG 召回意图、用户意图。
失败/超时/输出非法时返回 None，由 ``QueryPlanner`` 保留规则兜底结果，
不阻断主对话。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import UUID

from agent.llm import (
    GenerationOptions,
    LLMMessage,
    LLMRequest,
    LLMRequestContext,
    LLMRole,
    LLMUsage,
    ModelPurpose,
    TextPart,
    response_text,
)
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from utils.ids import new_request_id
from utils.serialization import extract_json_object

from .models import QueryPlan, QueryPlanSource, RagRecallIntent, parse_rag_intent
from .rules import RuleResult

__all__ = ["LLMQueryPlanner", "QueryPlanResult"]

logger = logging.getLogger(__name__)


class QueryPlanResult:
    """一次规划的结果与可观测信息；供主循环写流程日志/累计 usage。"""

    __slots__ = ("plan", "usage", "latency_ms", "model_id", "prompt_text", "raw_output")

    def __init__(
        self,
        plan: QueryPlan,
        *,
        usage: LLMUsage | None = None,
        latency_ms: int | None = None,
        model_id: str | None = None,
        prompt_text: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        """保存规划产物与调用元信息。"""
        self.plan = plan
        self.usage = usage or LLMUsage()
        self.latency_ms = latency_ms
        self.model_id = model_id
        self.prompt_text = prompt_text
        self.raw_output = raw_output


class LLMQueryPlanner:
    """基于 Agent LLM 端口的查询规划器。"""

    def __init__(
        self,
        llm: Any,
        *,
        model_id: str,
        timeout_s: float = 8.0,
        max_output_tokens: int = 512,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存 LLM 端口、模型与超时；不发起网络请求。"""
        self.llm = llm
        self.model_id = model_id
        self.timeout_s = max(0.1, float(timeout_s))
        self.max_output_tokens = max(32, int(max_output_tokens))
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.AGENT_QUERY_PLAN_SYSTEM
        )

    async def plan(
        self,
        original_query: str,
        *,
        normalized_query: str,
        profile_text: str | None,
        session_summary: str | None,
        history_text: str | None,
        domain_catalog: str | None,
        rule: RuleResult,
        trace_id: str,
        conversation_id: UUID,
        turn_id: str,
        user_id: UUID,
    ) -> QueryPlanResult | None:
        """调用 LLM 生成规划；失败/超时/非法输出返回 None。"""
        prompt_text = _build_user_prompt(
            original_query=original_query,
            profile_text=profile_text,
            session_summary=session_summary,
            history_text=history_text,
            domain_catalog=domain_catalog,
            rule=rule,
        )
        request = LLMRequest(
            request_id=new_request_id(),
            context=LLMRequestContext(
                trace_id=trace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                user_id=user_id,
                purpose=ModelPurpose.QUERY_PLAN,
            ),
            model_id=self.model_id,
            messages=[
                LLMMessage(
                    role=LLMRole.SYSTEM,
                    content=[TextPart(text=self.system_prompt)],
                ),
                LLMMessage(
                    role=LLMRole.USER,
                    content=[TextPart(text=prompt_text)],
                ),
            ],
            options=GenerationOptions(
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.llm.chat(request),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("query 规划超时: model=%s", self.model_id)
            return None
        except Exception as exc:  # noqa: BLE001 - 规划可降级
            logger.warning("query 规划失败: %s", exc)
            return None
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw = response_text(response)
        plan = _parse_plan(
            raw,
            original_query=original_query,
            normalized_query=normalized_query,
        )
        if plan is None:
            logger.warning("query 规划输出无法解析: %r", raw[:200])
            return None
        return QueryPlanResult(
            plan,
            usage=getattr(response, "usage", None) or LLMUsage(),
            latency_ms=latency_ms,
            model_id=self.model_id,
            prompt_text=prompt_text,
            raw_output=raw,
        )


def _build_user_prompt(
    *,
    original_query: str,
    profile_text: str | None,
    session_summary: str | None,
    history_text: str | None,
    domain_catalog: str | None,
    rule: RuleResult,
) -> str:
    """把画像/历史/领域目录/原话/规则结论渲染成一次 user 消息。"""
    parts = [
        "## 用户画像\n" + (profile_text.strip() if profile_text else "（无）"),
    ]
    if session_summary and session_summary.strip():
        parts.append("## 会话摘要\n" + session_summary.strip())
    parts.append(
        "## 当前会话历史\n" + (history_text.strip() if history_text else "（无）")
    )
    parts.append(
        "## 垂直领域目录\n" + (domain_catalog.strip() if domain_catalog else "（无）")
    )
    parts.append("## 用户原话\n" + original_query.strip())
    parts.append(
        "## 规则初筛\n"
        f"needs_rag={rule.needs_rag if rule.needs_rag is not None else 'unknown'}; "
        f"rag_intent={rule.rag_intent.value if rule.rag_intent else 'none'}; "
        f"reason={rule.reason or '-'}"
    )
    parts.append("请按 system 要求只输出 JSON。")
    return "\n\n".join(parts)


def _parse_plan(
    raw: str,
    *,
    original_query: str,
    normalized_query: str,
) -> QueryPlan | None:
    """解析规划 JSON；字段缺失时按保守默认值补齐。"""
    payload = extract_json_object(raw)
    if payload is None:
        return None
    needs_rag = _as_bool(payload.get("needs_rag"))
    if needs_rag is None:
        return None
    rewritten = str(payload.get("rewritten_query") or "").strip()
    intent = parse_rag_intent(payload.get("rag_intent"))
    entities = _parse_entities(payload.get("entities"))
    if not needs_rag:
        rewritten = None
        intent = None
        entities = []
    else:
        rewritten = rewritten or normalized_query
        intent = intent or RagRecallIntent.LOOKUP
    user_intent = str(payload.get("user_intent") or "").strip() or normalized_query
    reason = str(payload.get("reason") or "").strip()
    return QueryPlan(
        original_query=original_query,
        normalized_query=normalized_query,
        needs_rag=needs_rag,
        rewritten_query=rewritten,
        rag_intent=intent,
        user_intent=user_intent,
        entities=entities,
        source=QueryPlanSource.LLM,
        confidence=0.85,
        reason=reason or "LLM 规划",
    )


def _parse_entities(value: Any) -> list[str]:
    """解析实体数组；去空、去重、限长，只保留字符串。"""
    if not isinstance(value, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = " ".join(str(item or "").split()).strip(" ,，、;；。")
        # 中文允许单字实体（如“犬”“猫”），英文至少两个字符。
        if (len(text) < 2 and text.isascii()) or len(text) > 60:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= 8:
            break
    return result


def _as_bool(value: Any) -> bool | None:
    """宽松解析布尔值；无法判断返回 None。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "需要", "是"}:
        return True
    if text in {"false", "no", "0", "不需要", "否"}:
        return False
    return None
