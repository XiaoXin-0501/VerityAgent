"""内置 ``resolve_chunk_conflict``：解决知识库片段之间的冲突。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agent.tools.enums import ToolKind, ToolSideEffect
from agent.tools.models import ToolContext, ToolError, ToolResult, ToolSpec
from agent.tools.policies import ToolPolicy
from agent.tools.registry import ToolRegistry
from exception.tool import ToolErrorCode

__all__ = ["ResolveConflictArgs", "register_conflict_tool"]


class ResolveConflictArgs(BaseModel):
    """片段冲突工具参数。"""

    # 至少两个冲突片段；LLM 从已召回片段中挑选
    chunk_ids: list[str] = Field(min_length=2, max_length=8)
    # 冲突性质：硬事实 / 观点差异 / 无法判断
    conflict_type: Literal["factual", "viewpoint", "ambiguous"] = "factual"
    # 用户原始问题/当前回答目标；工具据此判断冲突是否相关
    question: str = Field(min_length=1, max_length=2000)
    # 为什么认为这些片段冲突（可选，便于审计）
    reason: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def _unique_chunks(self) -> "ResolveConflictArgs":
        """冲突片段去重后仍至少两个。"""
        if len(set(self.chunk_ids)) < 2:
            raise ValueError("chunk_ids 至少需要两个不同片段")
        return self


async def _executor(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
    """调用冲突服务，返回可直接用于回答的结论。"""
    parsed = ResolveConflictArgs.model_validate(arguments)
    service = getattr(context.services, "chunk_conflict_service", None)
    if service is None:
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name="resolve_chunk_conflict",
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库冲突处理服务未配置",
                retryable=False,
            ),
        )
    try:
        chunk_ids = list(dict.fromkeys(parsed.chunk_ids))
        result = await service.resolve(
            user_id=context.user_id,
            permissions=context.permissions,
            question=parsed.question,
            conflict_type=parsed.conflict_type,
            chunk_ids=chunk_ids,
        )
    except Exception as exc:  # noqa: BLE001 - 工具错误统一结构化
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name="resolve_chunk_conflict",
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库冲突处理失败，请稍后重试",
                retryable=True,
                cause=str(exc),
            ),
        )
    answer = str(result.get("answer") or "").strip()
    if not answer:
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name="resolve_chunk_conflict",
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库冲突处理未返回结论",
                retryable=True,
            ),
        )
    return ToolResult.success(
        call_id=context.call_id,
        tool_name="resolve_chunk_conflict",
        content=answer,
        data={
            "conflict_type": parsed.conflict_type,
            "resolution": result.get("resolution"),
            "preferred_chunk_ids": result.get("preferred_chunk_ids") or [],
            "conflict_summary": result.get("conflict_summary") or "",
            "chunk_ids": chunk_ids,
        },
    )


def register_conflict_tool(registry: ToolRegistry) -> None:
    """注册片段冲突处理工具；只有配置显式开启时才暴露。"""
    registry.register_tool(
        ToolSpec(
            name="resolve_chunk_conflict",
            description=(
                "处理知识库片段之间的矛盾。检索结果中两个片段对同一问题给出"
                "冲突信息且会影响回答时，从检索结果里的 chunk_id 取至少两个"
                "片段，并传入冲突类型（factual/viewpoint/ambiguous）和用户问题；"
                "工具会结合文档时间、质量分与相邻片段给出结论。不要用它处理普通追问。"
            ),
            group="kb",
            kind=ToolKind.EXECUTABLE,
            side_effect=ToolSideEffect.READ,
            policy=ToolPolicy(),
            tags=["rag", "conflict"],
        ),
        _executor,
        argument_model=ResolveConflictArgs,
        source="builtin",
    )
