"""内置 ``resolve_chunk_conflict``：记录 chunk/fact 冲突并投递置信度惩罚。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from agent.tools.enums import ToolKind, ToolSideEffect
from agent.tools.models import ToolContext, ToolError, ToolResult, ToolSpec
from agent.tools.policies import ToolPolicy
from agent.tools.registry import ToolRegistry
from agent.workers.penalty import FactConfidencePenalty
from exception.tool import ToolErrorCode

__all__ = ["ResolveConflictArgs", "register_conflict_tool"]


class ResolveConflictArgs(BaseModel):
    """冲突工具参数；只影响事实置信度，不改 chunk 正文。"""

    fact_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    doc_ids: list[str] = Field(default_factory=list)
    delta: float = Field(default=0.1, gt=0.0, le=1.0)
    reason: str = ""

    @model_validator(mode="after")
    def _has_target(self) -> "ResolveConflictArgs":
        """冲突必须至少指向一类对象，否则惩罚没有作用对象。"""
        if not self.fact_ids and not self.chunk_ids and not self.doc_ids:
            raise ValueError("fact_ids/chunk_ids/doc_ids 至少提供一个")
        return self


async def _executor(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    """把冲突惩罚写入 Redis 队列，由 penalty worker 异步执行。"""
    parsed = ResolveConflictArgs.model_validate(arguments)
    queue = getattr(context.services, "conflict_penalty_queue", None)
    if queue is None:
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name="resolve_chunk_conflict",
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="冲突惩罚队列未配置",
                retryable=False,
            ),
        )
    await queue.enqueue(
        FactConfidencePenalty(
            user_id=context.user_id,
            fact_ids=list(parsed.fact_ids),
            delta=parsed.delta,
            reason=parsed.reason,
            chunk_ids=list(parsed.chunk_ids),
            doc_ids=list(parsed.doc_ids),
        )
    )
    return ToolResult.success(
        call_id=context.call_id,
        tool_name="resolve_chunk_conflict",
        content="冲突已记录，事实置信度将在后台调整。",
        data={"queued": True},
    )


def register_conflict_tool(registry: ToolRegistry) -> None:
    """注册冲突记录工具；只有配置显式开启时才暴露。"""
    registry.register_tool(
        ToolSpec(
            name="resolve_chunk_conflict",
            description=(
                "记录知识库片段与用户事实的冲突，并建议下调事实置信度。"
                "不修改 chunk、文档描述或事实正文。"
            ),
            group="kb",
            kind=ToolKind.EXECUTABLE,
            # 只投递异步惩罚，不直接改 chunk/文档/事实正文，因此按 READ 允许自动执行。
            # 重复调用会重复下调 confidence，worker 负责下限保护。
            side_effect=ToolSideEffect.READ,
            policy=ToolPolicy(),
            tags=["rag", "memory", "conflict"],
        ),
        _executor,
        argument_model=ResolveConflictArgs,
        source="builtin",
    )
