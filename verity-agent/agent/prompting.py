"""LLM Prompt 组装：把 ContextPlan/消息/工具投影成 LLMRequest。"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from agent.llm import (
    LLMMessage,
    LLMRequest,
    LLMRequestContext,
    LLMToolDescriptor,
    ModelPurpose,
    ToolChoice,
)
from utils.ids import new_request_id

__all__ = ["PromptBuilder"]


class PromptBuilder:
    """无状态 Prompt 组装器；不负责消息持久化/模型调用。"""

    def __init__(self, *, default_model_id: str) -> None:
        """保存默认模型 id。"""
        self.default_model_id = default_model_id

    def build(
        self,
        *,
        trace_id: str,
        conversation_id: UUID,
        turn_id: str,
        user_id: UUID,
        messages: list[LLMMessage],
        tools: list[LLMToolDescriptor],
        model_id: str | None = None,
        tool_choice: ToolChoice | None = None,
        purpose: ModelPurpose = ModelPurpose.CHAT,
        deadline: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> LLMRequest:
        """构造应用层 LLMRequest；请求 id 每次调用重新生成。"""
        return LLMRequest(
            request_id=new_request_id(),
            context=LLMRequestContext(
                trace_id=trace_id,
                conversation_id=conversation_id,
                turn_id=turn_id,
                user_id=user_id,
                purpose=purpose,
                deadline=deadline,
                metadata=dict(metadata or {}),
            ),
            model_id=model_id or self.default_model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice or ToolChoice(),
            stream=stream,
        )
