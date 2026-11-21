"""内置 ``resolve_chunk_conflict``：解决知识库片段之间的冲突。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agent.tools.declaration import ToolBuildContext, ToolDeclaration
from agent.tools.factory import SimpleToolFactory
from agent.tools.models import ToolContext, ToolError, ToolResult
from agent.tools.services import ToolServiceKey, get_tool_service
from exception.tool import ToolErrorCode

__all__ = ["ResolveConflictArgs", "ChunkConflictFactory", "execute_conflict_tool"]


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


async def execute_conflict_tool(
    arguments: dict[str, Any],
    context: ToolContext,
) -> ToolResult:
    """调用冲突服务，返回可直接用于回答的结论。"""
    parsed = ResolveConflictArgs.model_validate(arguments)
    service = get_tool_service(
        context.services, ToolServiceKey.CHUNK_CONFLICT_SERVICE
    )
    if service is None:
        return ToolResult.failure(
            call_id=context.call_id,
            tool_name=context.tool_name,
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
            tool_name=context.tool_name,
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
            tool_name=context.tool_name,
            error=ToolError(
                code=ToolErrorCode.UPSTREAM_ERROR,
                message="知识库冲突处理未返回结论",
                retryable=True,
            ),
        )
    return ToolResult.success(
        call_id=context.call_id,
        tool_name=context.tool_name,
        content=answer,
        data={
            "conflict_type": parsed.conflict_type,
            "resolution": result.get("resolution"),
            "preferred_chunk_ids": result.get("preferred_chunk_ids") or [],
            "conflict_summary": result.get("conflict_summary") or "",
            "chunk_ids": chunk_ids,
        },
    )


class ChunkConflictFactory(SimpleToolFactory):
    """``resolve_chunk_conflict`` 工厂：依赖冲突服务。"""

    argument_model = ResolveConflictArgs

    def create_executor(
        self,
        declaration: ToolDeclaration,
        ctx: ToolBuildContext,
    ):
        """冲突执行器是无状态函数。"""
        return execute_conflict_tool
