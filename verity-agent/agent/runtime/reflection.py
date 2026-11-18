"""主循环反思：每 N 次循环检查方向，必要时调整或止损。

反思是一次独立的短 LLM 调用，不参与主对话历史、不调用工具：

- 输入：用户原话、查询规划、已执行工具调用及结果摘要、当前迭代/预算；
- 输出：``continue`` / ``adjust`` / ``stop``、原因与可执行指导；
- ``adjust`` 时把指导注入下一轮主 LLM 请求；
- ``stop`` 时清空工具并强制最终回答，避免继续消耗预算或陷入无效循环；
- 反思失败/超时/输出非法时按 ``continue`` 处理，不阻断主流程。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import StrEnum
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
from config.agent import ReflectionConfig
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from utils.ids import new_request_id
from utils.serialization import extract_json_object

from .models import AgentLimits

__all__ = [
    "ReflectionAction",
    "ReflectionDecision",
    "ReflectionEngine",
    "ReflectionResult",
]

logger = logging.getLogger(__name__)


class ReflectionAction(StrEnum):
    """反思建议的下一步动作。"""

    CONTINUE = "continue"
    ADJUST = "adjust"
    STOP = "stop"


class ReflectionDecision:
    """一次反思的解析结果。"""

    __slots__ = ("direction_ok", "action", "reason", "guidance")

    def __init__(
        self,
        *,
        direction_ok: bool,
        action: ReflectionAction,
        reason: str = "",
        guidance: str = "",
    ) -> None:
        """保存反思结论。"""
        self.direction_ok = direction_ok
        self.action = action
        self.reason = reason
        self.guidance = guidance


class ReflectionResult:
    """反思调用结果与可观测信息。"""

    __slots__ = (
        "decision",
        "usage",
        "latency_ms",
        "model_id",
        "prompt_text",
        "raw_output",
    )

    def __init__(
        self,
        decision: ReflectionDecision,
        *,
        usage: LLMUsage | None = None,
        latency_ms: int | None = None,
        model_id: str | None = None,
        prompt_text: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        """保存结论与调用元信息。"""
        self.decision = decision
        self.usage = usage or LLMUsage()
        self.latency_ms = latency_ms
        self.model_id = model_id
        self.prompt_text = prompt_text
        self.raw_output = raw_output


class ReflectionEngine:
    """基于 Agent LLM 端口的反思器。"""

    def __init__(
        self,
        llm: Any,
        *,
        model_id: str,
        config: ReflectionConfig | None = None,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存 LLM 端口、模型与配置；不发起网络请求。"""
        self.llm = llm
        self.model_id = model_id
        self.config = config or ReflectionConfig()
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.AGENT_REFLECTION_SYSTEM
        )

    async def reflect(
        self,
        *,
        original_query: str,
        query_plan: Any,
        tool_history: list[dict[str, Any]],
        step_no: int,
        limits: AgentLimits,
        elapsed_ms: int,
        deadline_remaining_s: float,
        trace_id: str,
        conversation_id: UUID,
        turn_id: str,
        user_id: UUID,
    ) -> ReflectionResult | None:
        """执行一次反思；失败/超时/非法输出返回 None。"""
        prompt_text = _build_prompt(
            original_query=original_query,
            query_plan=query_plan,
            tool_history=tool_history,
            step_no=step_no,
            limits=limits,
            elapsed_ms=elapsed_ms,
            deadline_remaining_s=deadline_remaining_s,
            max_result_chars=self.config.max_tool_result_chars,
        )
        request = LLMRequest(
            request_id=new_request_id(),
            context=LLMRequestContext(
                trace_id=trace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                user_id=user_id,
                purpose=ModelPurpose.REFLECTION,
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
                max_output_tokens=self.config.max_output_tokens,
            ),
        )
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self.llm.chat(request),
                timeout=self.config.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("主循环反思超时: model=%s", self.model_id)
            return None
        except Exception as exc:  # noqa: BLE001 - 反思可降级
            logger.warning("主循环反思失败: %s", exc)
            return None
        latency_ms = int((time.perf_counter() - started) * 1000)
        raw = response_text(response)
        decision = _parse_decision(raw)
        if decision is None:
            logger.warning("主循环反思输出无法解析: %r", raw[:200])
            return None
        return ReflectionResult(
            decision,
            usage=getattr(response, "usage", None) or LLMUsage(),
            latency_ms=latency_ms,
            model_id=self.model_id,
            prompt_text=prompt_text,
            raw_output=raw,
        )


def _build_prompt(
    *,
    original_query: str,
    query_plan: Any,
    tool_history: list[dict[str, Any]],
    step_no: int,
    limits: AgentLimits,
    elapsed_ms: int,
    deadline_remaining_s: float,
    max_result_chars: int,
) -> str:
    """把用户问题、规划、工具进展和预算渲染成反思输入。"""
    lines = [
        "## 用户原话",
        original_query.strip(),
        "## 查询规划",
        f"needs_rag={getattr(query_plan, 'needs_rag', False)}; "
        f"rewritten_query={getattr(query_plan, 'rewritten_query', None) or '-'}; "
        f"rag_intent={getattr(query_plan, 'rag_intent_label', None) or '-'}; "
        f"user_intent={getattr(query_plan, 'user_intent', None) or '-'}",
        "## 已执行工具",
    ]
    if not tool_history:
        lines.append("（尚未调用工具）")
    else:
        for index, item in enumerate(tool_history, start=1):
            content = str(item.get("content") or "")
            if len(content) > max_result_chars:
                content = content[:max_result_chars] + "…[截断]"
            lines.append(
                f"{index}. tool={item.get('tool_name') or '-'} "
                f"status={item.get('status') or '-'} "
                f"耗时={item.get('duration_ms') or 0}ms "
                f"参数={_compact(item.get('arguments'))} "
                f"结果={content or '-'}"
            )
    lines.extend(
        [
            "## 运行状态",
            f"循环第 {step_no} 次；工具轮上限={limits.max_tool_rounds}；"
            f"LLM 调用上限={limits.max_llm_calls}；"
            f"已耗时={elapsed_ms}ms；剩余 deadline={int(max(0, deadline_remaining_s) * 1000)}ms",
            "请按 system 要求只输出 JSON。",
        ]
    )
    return "\n".join(lines)


def _compact(value: Any) -> str:
    """参数压缩成单行 JSON，最多 500 字符。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:500]


def _parse_decision(raw: str) -> ReflectionDecision | None:
    """解析反思 JSON；action 非法时返回 None。"""
    payload = extract_json_object(raw)
    if payload is None:
        return None
    action_text = str(payload.get("action") or "").strip().lower()
    try:
        action = ReflectionAction(action_text)
    except ValueError:
        return None
    direction_ok = payload.get("direction_ok")
    if not isinstance(direction_ok, bool):
        direction_ok = action is ReflectionAction.CONTINUE
    return ReflectionDecision(
        direction_ok=direction_ok,
        action=action,
        reason=str(payload.get("reason") or "").strip(),
        guidance=str(payload.get("guidance") or "").strip(),
    )
