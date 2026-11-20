"""UserFact 冲突判定：规则优先，必要时 LLM 兜底。"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import BaseModel, ConfigDict
from prompt import PromptId, PromptRegistry, get_default_prompt_registry
from provider.enums import ModelRole
from provider.llm.helpers import response_text as _response_text
from provider.llm.types import (
    ChatModelRequest,
    ModelGenerationOptions,
    ModelMessage,
    ModelTextPart,
)
from utils.serialization import parse_json_object
from utils.text import normalize_whitespace

__all__ = ["FactConflictDecision", "FactConflictService"]

logger = logging.getLogger(__name__)


class FactConflictDecision(BaseModel):
    """冲突判定结果。

    - ``direct_override``：同一主体/属性的明确正反冲突，后出现的事实覆盖旧事实；
    - ``merge``：同一偏好维度的不同取值，可合并为一条更完整的事实；
    - ``partial_conflict``：旧事实部分被否定，需要同时改写旧事实并新增一条事实；
    - ``duplicate/compatible/supersede/unresolved``：兼容旧语义。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: str = "compatible"
    target_id: str | None = None
    # 改写后的现有事实内容（direct_override / merge / partial_conflict）
    merged_content: str | None = None
    # 部分冲突时额外产生的新事实（partial_conflict）
    new_content: str | None = None
    reason: str = ""


class FactConflictService:
    """对新事实做冲突判定；LLM 失败时保守返回 compatible。"""

    def __init__(
        self,
        *,
        model: Any | None,
        model_id: str,
        max_candidates: int = 10,
        enabled: bool = True,
        timeout_s: float = 20.0,
        max_output_tokens: int = 128,
        prompts: PromptRegistry | None = None,
    ) -> None:
        """保存判定模型、Prompt 注册表与策略。"""
        self.model = model
        self.model_id = model_id
        self.max_candidates = max(1, int(max_candidates))
        self.enabled = enabled
        self.timeout_s = max(1.0, float(timeout_s))
        self.max_output_tokens = max(16, int(max_output_tokens))
        self.prompts = prompts or get_default_prompt_registry()
        self.system_prompt = self.prompts.render_text(
            PromptId.MEMORY_FACT_CONFLICT_SYSTEM
        )

    async def decide(
        self,
        *,
        new_fact: dict[str, Any],
        candidates: list[dict[str, Any]],
    ) -> FactConflictDecision:
        """规则匹配完全相同内容，其余交给 LLM；失败按 compatible。"""
        if not self.enabled or not candidates:
            return FactConflictDecision()
        normalized = _normalize(str(new_fact.get("content") or ""))
        for candidate in candidates[: self.max_candidates]:
            if _normalize(str(candidate.get("content") or "")) == normalized:
                return FactConflictDecision(
                    decision="duplicate",
                    target_id=str(candidate.get("id") or ""),
                    reason="normalized content 相同",
                )
        if self.model is None:
            return FactConflictDecision()
        payload = {
            "new_fact": new_fact,
            "candidates": candidates[: self.max_candidates],
        }
        request = ChatModelRequest(
            model=self.model_id,
            messages=[
                ModelMessage(
                    role=ModelRole.SYSTEM,
                    content=[ModelTextPart(text=self.system_prompt)],
                ),
                ModelMessage(
                    role=ModelRole.USER,
                    content=[
                        ModelTextPart(
                            text=json.dumps(payload, ensure_ascii=False)
                        )
                    ],
                ),
            ],
            options=ModelGenerationOptions(
                temperature=0.0,
                max_output_tokens=self.max_output_tokens,
            ),
        )
        try:
            response = await asyncio.wait_for(
                self.model.complete(request),
                timeout=self.timeout_s,
            )
            data = parse_json_object(_response_text(response))
            decision = str(data.get("decision") or "compatible").lower()
            if decision not in {
                "compatible",
                "duplicate",
                "supersede",
                "direct_override",
                "merge",
                "partial_conflict",
                "unresolved",
            }:
                decision = "compatible"
            merged_content = _clip(str(data.get("merged_content") or ""))
            new_content = _clip(str(data.get("new_content") or ""))
            return FactConflictDecision(
                decision=decision,
                target_id=(str(data.get("target_id")) if data.get("target_id") else None),
                merged_content=merged_content or None,
                new_content=new_content or None,
                reason=str(data.get("reason") or "")[:300],
            )
        except Exception:  # noqa: BLE001 - 冲突判定失败保守处理
            logger.warning("事实冲突判定失败，按 compatible 处理", exc_info=True)
            return FactConflictDecision()


def _normalize(value: str) -> str:
    """规范化事实文本用于精确去重。"""
    return normalize_whitespace(value, keep_newlines=False).lower()


def _clip(value: str, limit: int = 300) -> str:
    """限制模型生成的事实文本长度。"""
    text = normalize_whitespace(value or "", keep_newlines=False).strip()
    return text[:limit]
